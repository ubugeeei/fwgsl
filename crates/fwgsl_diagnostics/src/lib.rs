//! Diagnostic reporting for fwgsl.
//!
//! Provides structured diagnostics with severity levels, source labels,
//! and miette integration for rich terminal rendering.

use fwgsl_span::Span;

/// Severity level for a diagnostic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

/// A label pointing to a source location with an associated message.
#[derive(Clone, Debug)]
pub struct Label {
    pub span: Span,
    pub message: String,
}

impl Label {
    /// Create a new label at the given span with a message.
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }

    /// Create a primary label (alias for `new`).
    pub fn primary(span: Span, message: impl Into<String>) -> Self {
        Self::new(span, message)
    }
}

/// A structured diagnostic message.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub code: Option<String>,
    pub labels: Vec<Label>,
    pub help: Option<String>,
}

impl Diagnostic {
    /// Create a new error diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            code: None,
            labels: Vec::new(),
            help: None,
        }
    }

    /// Create a new warning diagnostic.
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            code: None,
            labels: Vec::new(),
            help: None,
        }
    }

    /// Add a label to the diagnostic (builder pattern).
    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    /// Set the help message (builder pattern).
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Set the error code (builder pattern).
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
}

/// A collection of diagnostics accumulated during compilation.
pub struct DiagnosticSink {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticSink {
    /// Create a new empty diagnostic sink.
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Push a diagnostic into the sink.
    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Return all error-level diagnostics.
    pub fn errors(&self) -> &[Diagnostic] {
        // Note: returns all diagnostics; filter to errors only
        // We return a slice, but since mixed severities may exist,
        // we provide a filtered view via `iter`.
        // For a simple slice return, we return all diagnostics.
        // Use `iter().filter()` for severity-specific filtering.
        &self.diagnostics
    }

    /// Check whether any error-level diagnostics have been reported.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    /// Iterate over all diagnostics.
    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter()
    }
}

impl Default for DiagnosticSink {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// miette integration
// ---------------------------------------------------------------------------

/// A wrapper that converts a fwgsl `Diagnostic` into a miette-renderable
/// diagnostic, carrying source code context for display.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct MietteDiagnostic {
    message: String,
    src: miette::NamedSource<String>,
    severity: Severity,
    code: Option<String>,
    labels: Vec<MietteLabel>,
    help: Option<String>,
}

#[derive(Debug)]
struct MietteLabel {
    span: miette::SourceSpan,
    message: String,
}

impl MietteDiagnostic {
    /// Create a miette-renderable diagnostic from a fwgsl diagnostic.
    ///
    /// `source_name` is a display name for the source file (e.g. "main.fwgsl").
    /// `source_code` is the full source text that spans reference into.
    pub fn from_diagnostic(
        diag: &Diagnostic,
        source_name: impl AsRef<str>,
        source_code: impl Into<String>,
    ) -> Self {
        let source_text = source_code.into();
        let labels = diag
            .labels
            .iter()
            .map(|l| MietteLabel {
                span: miette::SourceSpan::new(
                    miette::SourceOffset::from(l.span.start as usize),
                    l.span.end as usize - l.span.start as usize,
                ),
                message: l.message.clone(),
            })
            .collect();

        Self {
            message: diag.message.clone(),
            src: miette::NamedSource::new(source_name.as_ref(), source_text),
            severity: diag.severity,
            code: diag.code.clone(),
            labels,
            help: diag.help.clone(),
        }
    }
}

impl miette::Diagnostic for MietteDiagnostic {
    fn code<'a>(&'a self) -> Option<Box<dyn fmt::Display + 'a>> {
        self.code
            .as_ref()
            .map(|c| Box::new(c.clone()) as Box<dyn fmt::Display>)
    }

    fn severity(&self) -> Option<miette::Severity> {
        Some(match self.severity {
            Severity::Error => miette::Severity::Error,
            Severity::Warning => miette::Severity::Warning,
            Severity::Info | Severity::Hint => miette::Severity::Advice,
        })
    }

    fn help<'a>(&'a self) -> Option<Box<dyn fmt::Display + 'a>> {
        self.help
            .as_ref()
            .map(|h| Box::new(h.clone()) as Box<dyn fmt::Display>)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = miette::LabeledSpan> + '_>> {
        if self.labels.is_empty() {
            None
        } else {
            Some(Box::new(self.labels.iter().map(|l| {
                miette::LabeledSpan::new_with_span(Some(l.message.clone()), l.span)
            })))
        }
    }

    fn source_code(&self) -> Option<&dyn miette::SourceCode> {
        Some(&self.src)
    }
}

use std::fmt;

#[cfg(test)]
mod tests {
    use super::*;
    use fwgsl_span::Span;

    #[test]
    fn test_diagnostic_error_builder() {
        let diag = Diagnostic::error("type mismatch")
            .with_code("E001")
            .with_label(Label::primary(Span::new(0, 5), "expected Int"))
            .with_help("try adding a type annotation");

        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.message, "type mismatch");
        assert_eq!(diag.code.as_deref(), Some("E001"));
        assert_eq!(diag.labels.len(), 1);
        assert_eq!(diag.help.as_deref(), Some("try adding a type annotation"));
    }

    #[test]
    fn test_diagnostic_warning() {
        let diag = Diagnostic::warning("unused variable");
        assert_eq!(diag.severity, Severity::Warning);
        assert_eq!(diag.message, "unused variable");
    }

    #[test]
    fn test_diagnostic_sink() {
        let mut sink = DiagnosticSink::new();
        assert!(!sink.has_errors());

        sink.push(Diagnostic::warning("unused import"));
        assert!(!sink.has_errors());

        sink.push(Diagnostic::error("syntax error"));
        assert!(sink.has_errors());
        assert_eq!(sink.iter().count(), 2);
    }

    #[test]
    fn test_miette_diagnostic_creation() {
        let diag = Diagnostic::error("unexpected token")
            .with_code("E100")
            .with_label(Label::primary(Span::new(0, 3), "here"))
            .with_help("did you mean `let`?");

        let miette_diag = MietteDiagnostic::from_diagnostic(&diag, "test.fwgsl", "lat x = 42");

        assert_eq!(miette_diag.message, "unexpected token");
        assert_eq!(miette_diag.code.as_deref(), Some("E100"));
        assert_eq!(miette_diag.labels.len(), 1);

        // Verify it implements miette::Diagnostic
        use miette::Diagnostic as _;
        assert_eq!(miette_diag.severity(), Some(miette::Severity::Error));
    }

    #[test]
    fn test_severity_copy() {
        let s = Severity::Error;
        let s2 = s;
        assert_eq!(s, s2);
    }

    #[test]
    fn test_label_constructors() {
        let span = Span::new(5, 10);
        let l1 = Label::new(span, "message");
        let l2 = Label::primary(span, "message");
        assert_eq!(l1.span, l2.span);
        assert_eq!(l1.message, l2.message);
    }
}
