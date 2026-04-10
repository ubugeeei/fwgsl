use std::sync::OnceLock;

use crate::parser::{Parser, Program};

const PRELUDE_SOURCE: &str = include_str!("../../../prelude/prelude.shadml");

static PRELUDE: OnceLock<Program> = OnceLock::new();

/// Returns the parsed prelude program (cached after first call).
pub fn prelude_program() -> &'static Program {
    PRELUDE.get_or_init(|| {
        let mut parser = Parser::new(PRELUDE_SOURCE);
        let program = parser.parse_program();
        assert!(
            !parser.diagnostics().has_errors(),
            "prelude parse errors: {:?}",
            parser.diagnostics().iter().collect::<Vec<_>>()
        );
        program
    })
}
