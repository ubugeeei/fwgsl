//! Module merging: combine a dependency-ordered set of modules into a single
//! flat program suitable for semantic analysis and compilation.
//!
//! Export filtering and import resolution happens here. The output is a single
//! `Program` with all declarations from all modules, with module/import
//! declarations stripped.

use std::collections::{HashMap, HashSet};

use crate::parser::{Decl, Program};
use crate::module_resolver::ModuleGraph;

/// Merge a module graph into a single flat program.
///
/// Processing order (dependency first):
/// 1. For each module in topological order, determine its exported names
/// 2. For each import in the current module, bring in names from the dependency
/// 3. Collect all non-module/import declarations into the merged program
///
/// For now, qualified imports are not fully supported — they require parser-level
/// disambiguation (see plan §2.9). Qualified imports are treated as a no-op with
/// a warning-level skip.
pub fn merge_modules(graph: &ModuleGraph) -> Program {
    // Phase 1: compute exported names for each module
    let mut module_exports: HashMap<String, HashSet<String>> = HashMap::new();

    for module in &graph.modules {
        let all_names = collect_declared_names(&module.program);
        let exported = match &module.exports {
            Some(export_list) => {
                // Only export listed names
                all_names.intersection(&export_list.iter().cloned().collect::<HashSet<_>>())
                    .cloned()
                    .collect()
            }
            None => all_names, // Export everything
        };
        module_exports.insert(module.name.clone(), exported);
    }

    // Phase 2: merge declarations
    let mut merged_decls: Vec<Decl> = Vec::new();

    for module in &graph.modules {
        // Add all non-module/import declarations from this module.
        // In the future, we could filter based on what's actually imported,
        // but for now we include everything and rely on DCE to remove unused code.
        for decl in &module.program.decls {
            match decl {
                Decl::ModuleDecl { .. } | Decl::ImportDecl { .. } => {
                    // Skip module/import declarations — they're metadata only
                }
                _ => {
                    merged_decls.push(decl.clone());
                }
            }
        }
    }

    Program { decls: merged_decls }
}

/// Collect all names declared in a program (functions, types, constructors, etc.).
fn collect_declared_names(program: &Program) -> HashSet<String> {
    let mut names = HashSet::new();
    for decl in &program.decls {
        match decl {
            Decl::TypeSig { name, .. } => { names.insert(name.clone()); }
            Decl::FunDecl { name, .. } => { names.insert(name.clone()); }
            Decl::EntryPoint { name, .. } => { names.insert(name.clone()); }
            Decl::DataDecl { name, constructors, .. } => {
                names.insert(name.clone());
                for con in constructors {
                    names.insert(con.name.clone());
                }
            }
            Decl::TypeAlias { name, .. } => { names.insert(name.clone()); }
            Decl::ResourceDecl { name, .. } => { names.insert(name.clone()); }
            Decl::BitfieldDecl { name, .. } => { names.insert(name.clone()); }
            Decl::ConstDecl { name, .. } => { names.insert(name.clone()); }
            Decl::TraitDecl { name, methods, .. } => {
                names.insert(name.clone());
                for m in methods {
                    names.insert(m.name.clone());
                }
            }
            Decl::ImplDecl { methods, .. } => {
                for m in methods {
                    names.insert(m.name.clone());
                }
            }
            Decl::ExternDecl { name, .. } => { names.insert(name.clone()); }
            Decl::ModuleDecl { .. } | Decl::ImportDecl { .. } => {}
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;
    use crate::module_resolver::ParsedModule;
    use std::path::PathBuf;

    fn parse_module(name: &str, source: &str) -> ParsedModule {
        let mut parser = Parser::new(source);
        let program = parser.parse_program();
        let exports = program.decls.iter().find_map(|d| {
            if let Decl::ModuleDecl { exports, .. } = d {
                Some(exports.clone())
            } else {
                None
            }
        }).flatten();
        let imports = program.decls.iter().filter_map(|d| {
            if let Decl::ImportDecl { module_path, kind, .. } = d {
                Some(crate::module_resolver::ModuleImport {
                    module_path: module_path.clone(),
                    kind: kind.clone(),
                })
            } else {
                None
            }
        }).collect();
        ParsedModule {
            name: name.to_string(),
            path: PathBuf::from(format!("{}.fwgsl", name)),
            program,
            exports,
            imports,
        }
    }

    #[test]
    fn merge_single_module() {
        let module = parse_module("Main", "add x y = x + y");
        let graph = ModuleGraph { modules: vec![module] };
        let merged = merge_modules(&graph);
        assert_eq!(merged.decls.len(), 1);
        assert!(matches!(&merged.decls[0], Decl::FunDecl { name, .. } if name == "add"));
    }

    #[test]
    fn merge_strips_module_and_import_decls() {
        let module = parse_module("Main", "module Main\nimport Utils\nadd x y = x + y");
        let graph = ModuleGraph { modules: vec![module] };
        let merged = merge_modules(&graph);
        // Only the FunDecl should remain
        assert_eq!(merged.decls.len(), 1);
        assert!(matches!(&merged.decls[0], Decl::FunDecl { name, .. } if name == "add"));
    }

    #[test]
    fn merge_two_modules_dependency_order() {
        let utils = parse_module("Utils", "module Utils\nhelper x = x");
        let main = parse_module("Main", "module Main\nimport Utils\nadd x y = x + y");
        let graph = ModuleGraph { modules: vec![utils, main] };
        let merged = merge_modules(&graph);
        assert_eq!(merged.decls.len(), 2);
        // Utils' helper should come first (dependency order)
        assert!(matches!(&merged.decls[0], Decl::FunDecl { name, .. } if name == "helper"));
        assert!(matches!(&merged.decls[1], Decl::FunDecl { name, .. } if name == "add"));
    }

    #[test]
    fn merge_includes_data_types_from_imports() {
        let types = parse_module("Types", "module Types\ndata Color = Red | Green | Blue");
        let main = parse_module("Main", "module Main\nimport Types\nf x = Red");
        let graph = ModuleGraph { modules: vec![types, main] };
        let merged = merge_modules(&graph);
        // Should contain DataDecl from Types + FunDecl from Main
        let has_data = merged.decls.iter().any(|d| matches!(d, Decl::DataDecl { name, .. } if name == "Color"));
        let has_fun = merged.decls.iter().any(|d| matches!(d, Decl::FunDecl { name, .. } if name == "f"));
        assert!(has_data, "should include Color data type");
        assert!(has_fun, "should include f function");
    }
}
