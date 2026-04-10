//! Virtual filesystem for browser/WASM and testing.
//!
//! Provides an in-memory `SourceReader` implementation and a bundle parser
//! for embedding multiple modules in a single string.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::module_resolver::SourceReader;

/// In-memory virtual filesystem.
///
/// Used for browser/WASM compilation, testing, and single-file bundles.
#[derive(Debug, Clone)]
pub struct VirtualFs {
    files: HashMap<PathBuf, String>,
}

impl VirtualFs {
    pub fn new() -> Self {
        VirtualFs {
            files: HashMap::new(),
        }
    }

    /// Add a file to the virtual filesystem.
    pub fn add(&mut self, path: impl Into<PathBuf>, source: impl Into<String>) {
        self.files.insert(path.into(), source.into());
    }

    /// List all files in the virtual filesystem.
    pub fn files(&self) -> impl Iterator<Item = (&Path, &str)> {
        self.files.iter().map(|(p, s)| (p.as_path(), s.as_str()))
    }

    /// List all .shadml files under a directory.
    pub fn list_dir(&self, dir: &Path) -> Vec<PathBuf> {
        self.files
            .keys()
            .filter(|p| {
                p.parent() == Some(dir) && p.extension().and_then(|s| s.to_str()) == Some("shadml")
            })
            .cloned()
            .collect()
    }
}

impl Default for VirtualFs {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceReader for VirtualFs {
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

/// Parse a bundle string containing multiple modules separated by
/// `--- module Name ---` markers.
///
/// Returns a `VirtualFs` with each module as a separate file.
/// Module names are mapped to paths: `Math.Vec` → `Math/Vec.shadml`.
///
/// Example input:
/// ```text
/// --- module Math.Vec ---
/// dot2 a b = a.x * b.x + a.y * b.y
///
/// --- module Main ---
/// import Math.Vec
/// f x = dot2 x x
/// ```
pub fn parse_bundle(source: &str) -> VirtualFs {
    let mut fs = VirtualFs::new();
    let mut current_module: Option<String> = None;
    let mut current_source = String::new();

    for line in source.lines() {
        if let Some(name) = parse_module_marker(line) {
            // Flush previous module
            if let Some(ref mod_name) = current_module {
                let path = module_name_to_path(mod_name);
                fs.add(path, current_source.trim().to_string());
                current_source.clear();
            }
            current_module = Some(name);
        } else if current_module.is_some() {
            current_source.push_str(line);
            current_source.push('\n');
        }
    }

    // Flush last module
    if let Some(ref mod_name) = current_module {
        let path = module_name_to_path(mod_name);
        fs.add(path, current_source.trim().to_string());
    }

    fs
}

/// Parse a `--- module Name ---` marker line.
fn parse_module_marker(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let prefix = "--- module ";
    let suffix = " ---";
    if trimmed.starts_with(prefix)
        && trimmed.ends_with(suffix)
        && trimmed.len() > prefix.len() + suffix.len()
    {
        let inner = trimmed[prefix.len()..trimmed.len() - suffix.len()].trim();
        if !inner.is_empty() {
            return Some(inner.to_string());
        }
    }
    None
}

/// Convert a dotted module name to a file path: `Math.Vec` → `Math/Vec.shadml`.
fn module_name_to_path(name: &str) -> PathBuf {
    let relative = name.replace('.', "/") + ".shadml";
    PathBuf::from(relative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_fs_basic() {
        let mut fs = VirtualFs::new();
        fs.add("Main.shadml", "f x = x");
        fs.add("Utils.shadml", "g x = x + 1");

        assert!(fs.file_exists(Path::new("Main.shadml")));
        assert!(!fs.file_exists(Path::new("Missing.shadml")));
        assert_eq!(fs.read_source(Path::new("Main.shadml")).unwrap(), "f x = x");
    }

    #[test]
    fn virtual_fs_list_dir() {
        let mut fs = VirtualFs::new();
        fs.add("Math/Vec.shadml", "dot2 a b = 0.0");
        fs.add("Math/Matrix.shadml", "identity = 1.0");
        fs.add("Main.shadml", "f x = x");

        let math_files = fs.list_dir(Path::new("Math"));
        assert_eq!(math_files.len(), 2);
    }

    #[test]
    fn parse_bundle_basic() {
        let bundle = r#"--- module Math.Vec ---
dot2 a b = a.x * b.x + a.y * b.y

--- module Main ---
import Math.Vec
f x = dot2 x x
"#;

        let fs = parse_bundle(bundle);
        assert!(fs.file_exists(Path::new("Math/Vec.shadml")));
        assert!(fs.file_exists(Path::new("Main.shadml")));

        let vec_src = fs.read_source(Path::new("Math/Vec.shadml")).unwrap();
        assert!(vec_src.contains("dot2"));

        let main_src = fs.read_source(Path::new("Main.shadml")).unwrap();
        assert!(main_src.contains("import Math.Vec"));
    }

    #[test]
    fn parse_bundle_empty() {
        let fs = parse_bundle("");
        assert_eq!(fs.files().count(), 0);
    }

    #[test]
    fn parse_bundle_single_module() {
        let bundle = "--- module Main ---\nf x = x\n";
        let fs = parse_bundle(bundle);
        assert_eq!(fs.files().count(), 1);
        assert!(fs.file_exists(Path::new("Main.shadml")));
    }

    #[test]
    fn module_marker_parsing() {
        assert_eq!(
            parse_module_marker("--- module Main ---"),
            Some("Main".to_string())
        );
        assert_eq!(
            parse_module_marker("--- module Math.Vec ---"),
            Some("Math.Vec".to_string())
        );
        assert_eq!(parse_module_marker("not a marker"), None);
        assert_eq!(parse_module_marker("--- module ---"), None);
    }
}
