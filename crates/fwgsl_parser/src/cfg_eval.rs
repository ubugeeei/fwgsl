//! Compile-time feature evaluation.
//!
//! Evaluates `cfg.name` predicates against a set of enabled features
//! and prunes inactive conditional declarations from the AST.

use std::collections::HashSet;

use crate::parser::{CfgPredicate, Decl, Program};

/// A set of enabled feature flags, typically from `--feature` CLI arguments.
#[derive(Debug, Clone)]
pub struct FeatureSet {
    features: HashSet<String>,
}

impl FeatureSet {
    /// Create an empty feature set (no features enabled).
    pub fn new() -> Self {
        FeatureSet { features: HashSet::new() }
    }

    /// Create a feature set from CLI flag strings.
    pub fn from_flags(flags: &[String]) -> Self {
        FeatureSet {
            features: flags.iter().cloned().collect(),
        }
    }

    /// Check if a feature is enabled.
    pub fn is_enabled(&self, name: &str) -> bool {
        self.features.contains(name)
    }

    /// Evaluate a cfg predicate against this feature set.
    pub fn evaluate(&self, pred: &CfgPredicate) -> bool {
        match pred {
            CfgPredicate::Feature(name) => self.features.contains(name.as_str()),
            CfgPredicate::Not(inner) => !self.evaluate(inner),
            CfgPredicate::And(a, b) => self.evaluate(a) && self.evaluate(b),
            CfgPredicate::Or(a, b) => self.evaluate(a) || self.evaluate(b),
        }
    }

    /// Get all enabled feature names.
    pub fn enabled_features(&self) -> &HashSet<String> {
        &self.features
    }
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate feature flags and prune inactive conditional declarations.
///
/// This modifies the program in-place:
/// - `Decl::CfgDecl` nodes are replaced with their active branch's declarations
/// - `ImportDecl` nodes with a false condition are removed
/// - All other declarations pass through unchanged
///
/// Returns the set of referenced feature names (for diagnostics).
pub fn evaluate_features(program: &mut Program, features: &FeatureSet) -> HashSet<String> {
    let mut referenced = HashSet::new();
    let old_decls = std::mem::take(&mut program.decls);
    program.decls = expand_decls(old_decls, features, &mut referenced);
    referenced
}

/// Recursively expand a list of declarations, evaluating cfg predicates.
fn expand_decls(
    decls: Vec<Decl>,
    features: &FeatureSet,
    referenced: &mut HashSet<String>,
) -> Vec<Decl> {
    let mut result = Vec::new();

    for decl in decls {
        match decl {
            Decl::CfgDecl { condition, then_decls, else_decls, .. } => {
                collect_referenced_features(&condition, referenced);
                if features.evaluate(&condition) {
                    // Recursively expand the then branch (may contain nested CfgDecls)
                    result.extend(expand_decls(then_decls, features, referenced));
                } else {
                    result.extend(expand_decls(else_decls, features, referenced));
                }
            }
            Decl::ImportDecl { ref condition, .. } => {
                if let Some(ref pred) = condition {
                    collect_referenced_features(pred, referenced);
                    if features.evaluate(pred) {
                        result.push(decl);
                    }
                    // else: conditional import with false predicate — drop it
                } else {
                    result.push(decl);
                }
            }
            _ => {
                result.push(decl);
            }
        }
    }

    result
}

/// Collect all feature names referenced in a predicate.
fn collect_referenced_features(pred: &CfgPredicate, referenced: &mut HashSet<String>) {
    match pred {
        CfgPredicate::Feature(name) => { referenced.insert(name.clone()); }
        CfgPredicate::Not(inner) => collect_referenced_features(inner, referenced),
        CfgPredicate::And(a, b) | CfgPredicate::Or(a, b) => {
            collect_referenced_features(a, referenced);
            collect_referenced_features(b, referenced);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn parse(source: &str) -> Program {
        let mut parser = Parser::new(source);
        parser.parse_program()
    }

    #[test]
    fn feature_set_basic() {
        let fs = FeatureSet::from_flags(&["debug".to_string(), "msaa".to_string()]);
        assert!(fs.is_enabled("debug"));
        assert!(fs.is_enabled("msaa"));
        assert!(!fs.is_enabled("mobile"));
    }

    #[test]
    fn evaluate_feature_predicate() {
        let fs = FeatureSet::from_flags(&["debug".to_string()]);

        assert!(fs.evaluate(&CfgPredicate::Feature("debug".into())));
        assert!(!fs.evaluate(&CfgPredicate::Feature("msaa".into())));
        assert!(fs.evaluate(&CfgPredicate::Not(Box::new(CfgPredicate::Feature("msaa".into())))));
        assert!(!fs.evaluate(&CfgPredicate::And(
            Box::new(CfgPredicate::Feature("debug".into())),
            Box::new(CfgPredicate::Feature("msaa".into())),
        )));
        assert!(fs.evaluate(&CfgPredicate::Or(
            Box::new(CfgPredicate::Feature("debug".into())),
            Box::new(CfgPredicate::Feature("msaa".into())),
        )));
    }

    #[test]
    fn evaluate_block_when_true() {
        let source = "when cfg.debug\n  debugVal = 1\nf x = x";
        let mut program = parse(source);
        let features = FeatureSet::from_flags(&["debug".to_string()]);
        evaluate_features(&mut program, &features);

        // debugVal should be present
        let names: Vec<_> = program.decls.iter().filter_map(|d| {
            match d {
                Decl::FunDecl { name, .. } => Some(name.as_str()),
                _ => None,
            }
        }).collect();
        assert!(names.contains(&"debugVal"));
        assert!(names.contains(&"f"));
    }

    #[test]
    fn evaluate_block_when_false() {
        let source = "when cfg.debug\n  debugVal = 1\nf x = x";
        let mut program = parse(source);
        let features = FeatureSet::new(); // debug not enabled
        evaluate_features(&mut program, &features);

        let names: Vec<_> = program.decls.iter().filter_map(|d| {
            match d {
                Decl::FunDecl { name, .. } => Some(name.as_str()),
                _ => None,
            }
        }).collect();
        assert!(!names.contains(&"debugVal"));
        assert!(names.contains(&"f"));
    }

    #[test]
    fn evaluate_conditional_import() {
        let source = "import Debug when cfg.debug\nf x = x";
        let mut program = parse(source);

        // With debug enabled
        let features = FeatureSet::from_flags(&["debug".to_string()]);
        let mut prog_on = program.clone();
        evaluate_features(&mut prog_on, &features);
        assert!(prog_on.decls.iter().any(|d| matches!(d, Decl::ImportDecl { .. })));

        // Without debug enabled
        let features = FeatureSet::new();
        evaluate_features(&mut program, &features);
        assert!(!program.decls.iter().any(|d| matches!(d, Decl::ImportDecl { .. })));
    }
}
