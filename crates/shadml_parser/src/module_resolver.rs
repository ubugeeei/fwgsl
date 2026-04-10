//! Module resolution for multi-file shadml projects.
//!
//! Given a root file and its parsed program, discovers all imported modules,
//! resolves their paths, parses them, and returns a dependency-ordered list
//! of modules suitable for compilation.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::parser::{Decl, ImportKind, Parser, Program};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A parsed module with its metadata.
#[derive(Debug, Clone)]
pub struct ParsedModule {
    /// Dotted module name (e.g. "Math.Fp64").
    pub name: String,
    /// File path this module was loaded from.
    pub path: PathBuf,
    /// The parsed AST.
    pub program: Program,
    /// What this module imports.
    pub imports: Vec<ModuleImport>,
}

/// A single import from a module.
#[derive(Debug, Clone)]
pub struct ModuleImport {
    pub module_path: String,
    pub kind: ImportKind,
}

/// The result of module resolution: a set of modules in dependency order.
#[derive(Debug)]
pub struct ModuleGraph {
    /// Modules in topological order (dependencies before dependents).
    /// The last entry is the root module.
    pub modules: Vec<ParsedModule>,
}

/// Errors that can occur during module resolution.
#[derive(Debug, Clone)]
pub enum ModuleResolveError {
    /// A module file could not be found.
    FileNotFound {
        module_path: String,
        searched: PathBuf,
    },
    /// Circular dependency detected.
    CyclicDependency { cycle: Vec<String> },
    /// Parse error in a module file.
    ParseError {
        module_path: String,
        file: PathBuf,
        messages: Vec<String>,
    },
    /// IO error reading a file.
    IoError { file: PathBuf, message: String },
}

impl std::fmt::Display for ModuleResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleResolveError::FileNotFound {
                module_path,
                searched,
            } => {
                write!(
                    f,
                    "module '{}' not found (searched {})",
                    module_path,
                    searched.display()
                )
            }
            ModuleResolveError::CyclicDependency { cycle } => {
                write!(f, "circular dependency: {}", cycle.join(" -> "))
            }
            ModuleResolveError::ParseError {
                module_path,
                file,
                messages,
            } => {
                write!(
                    f,
                    "parse errors in module '{}' ({}):",
                    module_path,
                    file.display()
                )?;
                for msg in messages {
                    write!(f, "\n  {}", msg)?;
                }
                Ok(())
            }
            ModuleResolveError::IoError { file, message } => {
                write!(f, "error reading {}: {}", file.display(), message)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// File system abstraction
// ---------------------------------------------------------------------------

/// Trait for reading source files. Enables testing without real file I/O.
pub trait SourceReader {
    fn read_source(&self, path: &Path) -> Result<String, String>;
    fn file_exists(&self, path: &Path) -> bool;
}

/// Default file system reader.
pub struct FsReader;

impl SourceReader for FsReader {
    fn read_source(&self, path: &Path) -> Result<String, String> {
        std::fs::read_to_string(path).map_err(|e| e.to_string())
    }
    fn file_exists(&self, path: &Path) -> bool {
        path.exists()
    }
}

// ---------------------------------------------------------------------------
// Module resolution
// ---------------------------------------------------------------------------

/// Resolve all modules starting from a root file.
///
/// - `root_path`: the path to the root .shadml file
/// - `root_program`: the already-parsed program from root_path
/// - `source_roots`: directories to search for module files (typically the
///   directory containing the root file)
/// - `reader`: source file reader
pub fn resolve_modules(
    root_path: &Path,
    root_program: Program,
    source_roots: &[PathBuf],
    reader: &dyn SourceReader,
) -> Result<ModuleGraph, Vec<ModuleResolveError>> {
    let mut resolver = Resolver::new(source_roots.to_vec(), reader);

    // Determine root module name from its module decl, or derive from file path
    let root_name = extract_module_name(&root_program)
        .unwrap_or_else(|| derive_module_name(root_path, source_roots));

    let root_imports = extract_imports(&root_program);

    let root_module = ParsedModule {
        name: root_name.clone(),
        path: root_path.to_path_buf(),
        program: root_program,
        imports: root_imports,
    };

    resolver.modules.insert(root_name.clone(), root_module);

    // Discover all transitive imports
    resolver.discover_imports(&root_name)?;

    // Topological sort
    let sorted = resolver.topological_sort(&root_name)?;

    Ok(ModuleGraph { modules: sorted })
}

// ---------------------------------------------------------------------------
// Internal resolver
// ---------------------------------------------------------------------------

struct Resolver<'a> {
    source_roots: Vec<PathBuf>,
    reader: &'a dyn SourceReader,
    modules: HashMap<String, ParsedModule>,
    errors: Vec<ModuleResolveError>,
}

impl<'a> Resolver<'a> {
    fn new(source_roots: Vec<PathBuf>, reader: &'a dyn SourceReader) -> Self {
        Resolver {
            source_roots,
            reader,
            modules: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Recursively discover and parse all imported modules.
    fn discover_imports(&mut self, module_name: &str) -> Result<(), Vec<ModuleResolveError>> {
        // Collect imports from this module
        let imports: Vec<ModuleImport> = self
            .modules
            .get(module_name)
            .map(|m| m.imports.clone())
            .unwrap_or_default();

        for import in imports {
            match &import.kind {
                ImportKind::Wildcard => {
                    // Wildcard: `import Foo.*` — discover all .shadml files under Foo/
                    let wildcard_modules = self.find_wildcard_modules(&import.module_path);
                    for (mod_name, file_path) in wildcard_modules {
                        if self.modules.contains_key(&mod_name) {
                            continue;
                        }
                        self.load_module(&mod_name, &file_path)?;
                        self.discover_imports(&mod_name)?;
                    }
                }
                _ => {
                    if self.modules.contains_key(&import.module_path) {
                        continue;
                    }

                    let file_path = match self.find_module_file(&import.module_path) {
                        Some(p) => p,
                        None => {
                            let searched = self
                                .source_roots
                                .first()
                                .cloned()
                                .unwrap_or_else(|| PathBuf::from("."));
                            self.errors.push(ModuleResolveError::FileNotFound {
                                module_path: import.module_path.clone(),
                                searched,
                            });
                            continue;
                        }
                    };

                    self.load_module(&import.module_path, &file_path)?;
                    self.discover_imports(&import.module_path)?;
                }
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    /// Load, parse, and register a single module.
    fn load_module(&mut self, name: &str, file_path: &Path) -> Result<(), Vec<ModuleResolveError>> {
        let source = match self.reader.read_source(file_path) {
            Ok(s) => s,
            Err(msg) => {
                self.errors.push(ModuleResolveError::IoError {
                    file: file_path.to_path_buf(),
                    message: msg,
                });
                return Ok(());
            }
        };

        let mut parser = Parser::new(&source);
        let program = parser.parse_program();

        if parser.diagnostics().has_errors() {
            let messages: Vec<String> = parser
                .diagnostics()
                .iter()
                .map(|d| d.message.clone())
                .collect();
            self.errors.push(ModuleResolveError::ParseError {
                module_path: name.to_string(),
                file: file_path.to_path_buf(),
                messages,
            });
            return Ok(());
        }

        let imports = extract_imports(&program);

        let parsed = ParsedModule {
            name: name.to_string(),
            path: file_path.to_path_buf(),
            program,
            imports,
        };

        self.modules.insert(name.to_string(), parsed);
        Ok(())
    }

    /// Find the file for a module path like "Math.Fp64" → "Math/Fp64.shadml".
    fn find_module_file(&self, module_path: &str) -> Option<PathBuf> {
        let relative = module_path.replace('.', "/") + ".shadml";
        for root in &self.source_roots {
            let candidate = root.join(&relative);
            if self.reader.file_exists(&candidate) {
                return Some(candidate);
            }
        }
        None
    }

    /// Find all .shadml files under a namespace directory for wildcard imports.
    /// `import Foo.*` looks for all `Foo/*.shadml` files in source roots.
    fn find_wildcard_modules(&self, namespace: &str) -> Vec<(String, PathBuf)> {
        let dir_relative = namespace.replace('.', "/");
        let mut results = Vec::new();
        for root in &self.source_roots {
            let dir = root.join(&dir_relative);
            // List all .shadml files in the directory
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("shadml") {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            let mod_name = format!("{}.{}", namespace, stem);
                            if !results
                                .iter()
                                .any(|(n, _): &(String, PathBuf)| n == &mod_name)
                            {
                                results.push((mod_name, path));
                            }
                        }
                    }
                }
            }
        }
        results
    }

    /// Topological sort of modules. Returns modules in dependency order.
    fn topological_sort(
        &self,
        root_name: &str,
    ) -> Result<Vec<ParsedModule>, Vec<ModuleResolveError>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut in_progress: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();
        let mut path_stack: Vec<String> = Vec::new();

        if let Err(cycle) = self.topo_visit(
            root_name,
            &mut visited,
            &mut in_progress,
            &mut order,
            &mut path_stack,
        ) {
            return Err(vec![ModuleResolveError::CyclicDependency { cycle }]);
        }

        let mut result = Vec::new();
        for name in &order {
            if let Some(module) = self.modules.get(name) {
                result.push(module.clone());
            }
        }

        Ok(result)
    }

    fn topo_visit(
        &self,
        name: &str,
        visited: &mut HashSet<String>,
        in_progress: &mut HashSet<String>,
        order: &mut Vec<String>,
        path_stack: &mut Vec<String>,
    ) -> Result<(), Vec<String>> {
        if visited.contains(name) {
            return Ok(());
        }
        if in_progress.contains(name) {
            // Found a cycle — build the cycle path
            let mut cycle: Vec<String> = path_stack.clone();
            cycle.push(name.to_string());
            return Err(cycle);
        }

        in_progress.insert(name.to_string());
        path_stack.push(name.to_string());

        if let Some(module) = self.modules.get(name) {
            for import in &module.imports {
                self.topo_visit(&import.module_path, visited, in_progress, order, path_stack)?;
            }
        }

        path_stack.pop();
        in_progress.remove(name);
        visited.insert(name.to_string());
        order.push(name.to_string());

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the module name from a `module Foo.Bar` declaration.
fn extract_module_name(program: &Program) -> Option<String> {
    for decl in &program.decls {
        if let Decl::ModuleDecl { name, .. } = decl {
            return Some(name.clone());
        }
    }
    None
}

/// Extract all import declarations from a program.
fn extract_imports(program: &Program) -> Vec<ModuleImport> {
    program
        .decls
        .iter()
        .filter_map(|decl| {
            if let Decl::ImportDecl {
                module_path, kind, ..
            } = decl
            {
                Some(ModuleImport {
                    module_path: module_path.clone(),
                    kind: kind.clone(),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Derive a module name from a file path relative to source roots.
/// `src/Math/Fp64.shadml` with source root `src/` → `Math.Fp64`.
/// Falls back to just the file stem if no source root matches.
fn derive_module_name(path: &Path, source_roots: &[PathBuf]) -> String {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        path.to_path_buf()
    };
    for root in source_roots {
        if let Ok(relative) = path.strip_prefix(root) {
            let name = relative
                .with_extension("")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join(".");
            if !name.is_empty() {
                return name;
            }
        }
    }
    // Fallback: just the file stem
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Main")
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory source reader for testing.
    struct MemoryReader {
        files: HashMap<PathBuf, String>,
    }

    impl MemoryReader {
        fn new() -> Self {
            MemoryReader {
                files: HashMap::new(),
            }
        }
        fn add(&mut self, path: impl Into<PathBuf>, source: &str) {
            self.files.insert(path.into(), source.to_string());
        }
    }

    impl SourceReader for MemoryReader {
        fn read_source(&self, path: &Path) -> Result<String, String> {
            self.files
                .get(path)
                .cloned()
                .ok_or_else(|| format!("file not found: {}", path.display()))
        }
        fn file_exists(&self, path: &Path) -> bool {
            self.files.contains_key(path)
        }
    }

    #[test]
    fn single_file_no_imports() {
        let source = "add x y = x + y";
        let mut parser = Parser::new(source);
        let program = parser.parse_program();

        let reader = MemoryReader::new();
        let root = PathBuf::from("/project/Main.shadml");
        let result = resolve_modules(&root, program, &[PathBuf::from("/project")], &reader);
        let graph = result.expect("should resolve");
        assert_eq!(graph.modules.len(), 1);
        assert_eq!(graph.modules[0].name, "Main");
    }

    #[test]
    fn two_modules() {
        let root_source = r#"module Main
import Utils
add x y = x + y
"#;
        let utils_source = r#"module Utils
helper x = x
"#;
        let mut parser = Parser::new(root_source);
        let root_program = parser.parse_program();

        let mut reader = MemoryReader::new();
        reader.add(PathBuf::from("/project/Utils.shadml"), utils_source);

        let root = PathBuf::from("/project/Main.shadml");
        let result = resolve_modules(&root, root_program, &[PathBuf::from("/project")], &reader);
        let graph = result.expect("should resolve");
        assert_eq!(graph.modules.len(), 2);
        // Utils should come before Main (dependency order)
        assert_eq!(graph.modules[0].name, "Utils");
        assert_eq!(graph.modules[1].name, "Main");
    }

    #[test]
    fn transitive_imports() {
        let main_source = r#"module Main
import A
f x = x
"#;
        let a_source = r#"module A
import B
g x = x
"#;
        let b_source = r#"module B
h x = x
"#;

        let mut parser = Parser::new(main_source);
        let root_program = parser.parse_program();

        let mut reader = MemoryReader::new();
        reader.add(PathBuf::from("/p/A.shadml"), a_source);
        reader.add(PathBuf::from("/p/B.shadml"), b_source);

        let root = PathBuf::from("/p/Main.shadml");
        let result = resolve_modules(&root, root_program, &[PathBuf::from("/p")], &reader);
        let graph = result.expect("should resolve");
        assert_eq!(graph.modules.len(), 3);
        // B before A before Main
        assert_eq!(graph.modules[0].name, "B");
        assert_eq!(graph.modules[1].name, "A");
        assert_eq!(graph.modules[2].name, "Main");
    }

    #[test]
    fn cycle_detection() {
        let a_source = r#"module A
import B
f x = x
"#;
        let b_source = r#"module B
import A
g x = x
"#;

        let mut parser = Parser::new(a_source);
        let root_program = parser.parse_program();

        let mut reader = MemoryReader::new();
        reader.add(PathBuf::from("/p/B.shadml"), b_source);
        reader.add(PathBuf::from("/p/A.shadml"), a_source);

        let root = PathBuf::from("/p/A.shadml");
        let result = resolve_modules(&root, root_program, &[PathBuf::from("/p")], &reader);
        match result {
            Err(errors) => {
                assert!(errors
                    .iter()
                    .any(|e| matches!(e, ModuleResolveError::CyclicDependency { .. })));
            }
            Ok(_) => panic!("expected cycle error"),
        }
    }

    #[test]
    fn missing_module() {
        let source = r#"module Main
import NonExistent
f x = x
"#;
        let mut parser = Parser::new(source);
        let root_program = parser.parse_program();

        let reader = MemoryReader::new();
        let root = PathBuf::from("/p/Main.shadml");
        let result = resolve_modules(&root, root_program, &[PathBuf::from("/p")], &reader);
        match result {
            Err(errors) => {
                assert!(errors
                    .iter()
                    .any(|e| matches!(e, ModuleResolveError::FileNotFound { .. })));
            }
            Ok(_) => panic!("expected file-not-found error"),
        }
    }

    #[test]
    fn nested_module_path() {
        let main_source = r#"module Main
import Math.Fp64
f x = x
"#;
        let fp64_source = r#"module Math.Fp64
data Fp64 = Fp64 F32 F32
"#;

        let mut parser = Parser::new(main_source);
        let root_program = parser.parse_program();

        let mut reader = MemoryReader::new();
        reader.add(PathBuf::from("/p/Math/Fp64.shadml"), fp64_source);

        let root = PathBuf::from("/p/Main.shadml");
        let result = resolve_modules(&root, root_program, &[PathBuf::from("/p")], &reader);
        let graph = result.expect("should resolve");
        assert_eq!(graph.modules.len(), 2);
        assert_eq!(graph.modules[0].name, "Math.Fp64");
        assert_eq!(graph.modules[1].name, "Main");
    }

    #[test]
    fn diamond_dependency() {
        // Main -> A, B; A -> C; B -> C
        let main_source = "module Main\nimport A\nimport B\nf x = x\n";
        let a_source = "module A\nimport C\ng x = x\n";
        let b_source = "module B\nimport C\nh x = x\n";
        let c_source = "module C\ni x = x\n";

        let mut parser = Parser::new(main_source);
        let root_program = parser.parse_program();

        let mut reader = MemoryReader::new();
        reader.add(PathBuf::from("/p/A.shadml"), a_source);
        reader.add(PathBuf::from("/p/B.shadml"), b_source);
        reader.add(PathBuf::from("/p/C.shadml"), c_source);

        let root = PathBuf::from("/p/Main.shadml");
        let result = resolve_modules(&root, root_program, &[PathBuf::from("/p")], &reader);
        let graph = result.expect("should resolve");
        assert_eq!(graph.modules.len(), 4);
        // C must come before both A and B; Main must be last
        let names: Vec<&str> = graph.modules.iter().map(|m| m.name.as_str()).collect();
        let c_pos = names.iter().position(|n| *n == "C").unwrap();
        let a_pos = names.iter().position(|n| *n == "A").unwrap();
        let b_pos = names.iter().position(|n| *n == "B").unwrap();
        let main_pos = names.iter().position(|n| *n == "Main").unwrap();
        assert!(c_pos < a_pos);
        assert!(c_pos < b_pos);
        assert!(a_pos < main_pos);
        assert!(b_pos < main_pos);
    }
}
