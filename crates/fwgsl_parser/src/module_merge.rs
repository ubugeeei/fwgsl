//! Module merging: combine a dependency-ordered set of modules into a single
//! flat program suitable for semantic analysis and compilation.
//!
//! All public (non-private) declarations are included. The output is a single
//! `Program` with all declarations from all modules, with module/import
//! declarations stripped.

use std::collections::HashSet;

use crate::parser::{Decl, Program};
use crate::module_resolver::ModuleGraph;

/// Merge a module graph into a single flat program.
///
/// Processing order (dependency first):
/// 1. For each module in topological order, collect all non-private,
///    non-module/import declarations into the merged program.
///
/// Visibility: everything is public by default. Declarations preceded by
/// `private` are excluded from the merged output (they stay module-local).
pub fn merge_modules(graph: &ModuleGraph) -> Program {
    let mut merged_decls: Vec<Decl> = Vec::new();

    for module in &graph.modules {
        let private_names = collect_private_names(&module.program);
        for decl in &module.program.decls {
            match decl {
                Decl::ModuleDecl { .. } | Decl::ImportDecl { .. } => {
                    // Skip module/import declarations — they're metadata only
                }
                _ => {
                    // Filter out private declarations
                    let name = decl_name(decl);
                    if let Some(n) = name {
                        if private_names.contains(n) {
                            continue;
                        }
                    }
                    merged_decls.push(decl.clone());
                }
            }
        }
    }

    Program { decls: merged_decls }
}

/// Get the primary name of a declaration, if any.
fn decl_name(decl: &Decl) -> Option<&str> {
    match decl {
        Decl::TypeSig { name, .. }
        | Decl::FunDecl { name, .. }
        | Decl::EntryPoint { name, .. }
        | Decl::DataDecl { name, .. }
        | Decl::TypeAlias { name, .. }
        | Decl::ResourceDecl { name, .. }
        | Decl::BitfieldDecl { name, .. }
        | Decl::ConstDecl { name, .. }
        | Decl::TraitDecl { name, .. }
        | Decl::ExternDecl { name, .. } => Some(name.as_str()),
        Decl::ImplDecl { .. }
        | Decl::ModuleDecl { .. }
        | Decl::ImportDecl { .. } => None,
    }
}

/// Collect names of declarations that follow a `private` keyword in the source.
///
/// For now, `private` is detected by scanning comments attached to declarations.
/// TODO: Once the parser emits a `Decl::Private` wrapper or flag, use that instead.
/// Currently this is a placeholder — full `private` support requires parser changes
/// to track which declarations are marked private.
fn collect_private_names(_program: &Program) -> HashSet<String> {
    // Placeholder: no declarations are private yet.
    // When the parser supports `private`, this will extract private names.
    HashSet::new()
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
