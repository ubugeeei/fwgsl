pub mod cfg_eval;
pub mod layout;
pub mod lexer;
pub mod module_merge;
pub mod module_resolver;
pub mod parser;
pub mod prelude;
pub mod virtual_fs;

pub use cfg_eval::{evaluate_features, FeatureSet};
pub use layout::resolve_layout;
pub use lexer::{lex, Token};
pub use module_merge::merge_modules;
pub use module_resolver::{resolve_modules, FsReader, ModuleGraph, ParsedModule, SourceReader};
pub use parser::{CfgPredicate, Parser, Program};
pub use prelude::prelude_program;
pub use virtual_fs::{parse_bundle, VirtualFs};
