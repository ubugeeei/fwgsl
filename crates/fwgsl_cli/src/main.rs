use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Collect --feature flags from anywhere in the args
    let features = collect_features(&args);

    match args.get(1).map(|s| s.as_str()) {
        Some("compile") | Some("c") => {
            let file = args.get(2).unwrap_or_else(|| {
                eprintln!("Usage: fwgsl compile <file.fwgsl>");
                process::exit(1);
            });
            cmd_compile(
                file,
                args.contains(&"--emit-ast".to_string()),
                args.contains(&"--preserve-comments".to_string()),
                &features,
            );
        }
        Some("check") => {
            let file = args.get(2).unwrap_or_else(|| {
                eprintln!("Usage: fwgsl check <file.fwgsl>");
                process::exit(1);
            });
            cmd_check(file, &features);
        }
        Some("fmt") => {
            let file = args.get(2).unwrap_or_else(|| {
                eprintln!("Usage: fwgsl fmt <file.fwgsl>");
                process::exit(1);
            });
            cmd_fmt(file);
        }
        Some("version") | Some("--version") | Some("-V") => {
            println!("fwgsl {}", env!("CARGO_PKG_VERSION"));
        }
        Some("help") | Some("--help") | Some("-h") | None => {
            print_help();
        }
        Some(cmd) => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!();
            print_help();
            process::exit(1);
        }
    }
}

fn print_help() {
    println!(
        r#"fwgsl - Pure functional language for WebGPU

USAGE:
    fwgsl <COMMAND> [OPTIONS]

COMMANDS:
    compile, c  <file>    Compile .fwgsl to .wgsl
    check       <file>    Type-check without emitting
    fmt         <file>    Format source code
    version               Print version
    help                  Print this help

OPTIONS:
    --emit-ast            Print AST debug output
    --preserve-comments   Preserve source comments in WGSL output
    --output, -o <file>   Output file (default: stdout)
    --feature <name>      Enable a compile-time feature flag (can be repeated)
"#
    );
}

/// Collect `--feature <name>` flags from the argument list.
fn collect_features(args: &[String]) -> Vec<String> {
    let mut features = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--feature" {
            if let Some(name) = iter.next() {
                features.push(name.clone());
            }
        }
    }
    features
}

fn with_prelude(program: &mut fwgsl_parser::parser::Program) {
    let prelude = fwgsl_parser::prelude_program();
    let mut combined = prelude.decls.clone();
    combined.append(&mut program.decls);
    program.decls = combined;
}

/// Check if the program has import declarations (needs multi-file resolution).
fn has_imports(program: &fwgsl_parser::parser::Program) -> bool {
    has_imports_in(&program.decls)
}

fn has_imports_in(decls: &[fwgsl_parser::parser::Decl]) -> bool {
    use fwgsl_parser::parser::Decl;
    decls.iter().any(|d| match d {
        Decl::ImportDecl { .. } => true,
        Decl::CfgDecl { then_decls, else_decls, .. } => {
            has_imports_in(then_decls) || has_imports_in(else_decls)
        }
        _ => false,
    })
}

fn cmd_compile(file: &str, emit_ast: bool, preserve_comments: bool, feature_flags: &[String]) {
    let source = read_file(file);

    // Parse the root file
    let mut parser = fwgsl_parser::parser::Parser::new(&source);
    let mut root_program = parser.parse_program();

    if emit_ast {
        println!("{:#?}", root_program);
        return;
    }

    // Check for parse errors
    if parser.diagnostics().has_errors() {
        print_diagnostics(
            &parser.diagnostics().iter().collect::<Vec<_>>(),
            file,
            &source,
        );
        process::exit(1);
    }

    // Evaluate feature flags (prune conditional declarations/imports)
    let features = fwgsl_parser::FeatureSet::from_flags(feature_flags);
    fwgsl_parser::evaluate_features(&mut root_program, &features);

    // If the program has imports, use the module resolver
    let mut program = if has_imports(&root_program) {
        let root_path = std::path::Path::new(file);
        let source_root = root_path.parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf();

        let reader = fwgsl_parser::FsReader;
        match fwgsl_parser::resolve_modules(root_path, root_program, &[source_root], &reader) {
            Ok(graph) => fwgsl_parser::merge_modules(&graph),
            Err(errors) => {
                for e in &errors {
                    eprintln!("error: {}", e);
                }
                process::exit(1);
            }
        }
    } else {
        root_program
    };

    // Prepend prelude declarations
    with_prelude(&mut program);

    // Semantic analysis
    let mut analyzer = fwgsl_semantic::SemanticAnalyzer::new();
    analyzer.analyze(&program);

    if analyzer.has_errors() {
        let diags: Vec<_> = analyzer.diagnostics().iter().collect();
        print_diagnostics(&diags, file, &source);
        process::exit(1);
    }

    // AST -> HIR lowering
    let mut lowering = fwgsl_ast_lowering::AstLowering::new(&analyzer);
    let hir = lowering.lower_program(&program);

    if lowering.has_errors() {
        let diags: Vec<_> = lowering.diagnostics().iter().collect();
        print_diagnostics(&diags, file, &source);
        process::exit(1);
    }

    // HIR -> MIR lowering
    let mir = match fwgsl_mir::lower::lower_hir_to_mir(&hir) {
        Ok(mir) => mir,
        Err(errors) => {
            for e in &errors {
                eprintln!("error: {}", e);
            }
            process::exit(1);
        }
    };

    // MIR -> WGSL codegen
    let wgsl = if preserve_comments {
        fwgsl_wgsl_codegen::emit_wgsl_with_comments(&mir)
    } else {
        fwgsl_wgsl_codegen::emit_wgsl(&mir)
    };

    println!(
        "// Generated by fwgsl compiler v{}",
        env!("CARGO_PKG_VERSION")
    );
    println!("// Source: {}", file);
    println!();
    print!("{}", wgsl);
}

fn cmd_check(file: &str, feature_flags: &[String]) {
    let source = read_file(file);

    let mut parser = fwgsl_parser::parser::Parser::new(&source);
    let mut root_program = parser.parse_program();

    let mut has_errors = false;

    if parser.diagnostics().has_errors() {
        print_diagnostics(
            &parser.diagnostics().iter().collect::<Vec<_>>(),
            file,
            &source,
        );
        has_errors = true;
    }

    // Evaluate feature flags (prune conditional declarations/imports)
    let features = fwgsl_parser::FeatureSet::from_flags(feature_flags);
    fwgsl_parser::evaluate_features(&mut root_program, &features);

    // If the program has imports, use the module resolver
    let mut program = if has_imports(&root_program) {
        let root_path = std::path::Path::new(file);
        let source_root = root_path.parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf();

        let reader = fwgsl_parser::FsReader;
        match fwgsl_parser::resolve_modules(root_path, root_program, &[source_root], &reader) {
            Ok(graph) => fwgsl_parser::merge_modules(&graph),
            Err(errors) => {
                for e in &errors {
                    eprintln!("error: {}", e);
                }
                process::exit(1);
            }
        }
    } else {
        root_program
    };

    with_prelude(&mut program);

    let mut analyzer = fwgsl_semantic::SemanticAnalyzer::new();
    analyzer.analyze(&program);

    if analyzer.has_errors() {
        let diags: Vec<_> = analyzer.diagnostics().iter().collect();
        print_diagnostics(&diags, file, &source);
        has_errors = true;
    }

    if has_errors {
        process::exit(1);
    } else {
        println!("No errors found in {}", file);
    }
}

fn cmd_fmt(file: &str) {
    let source = read_file(file);
    let formatted = fwgsl_formatter::format_default(&source);
    print!("{}", formatted);
}

fn read_file(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path, e);
        process::exit(1);
    })
}

fn print_diagnostics(diagnostics: &[&fwgsl_diagnostics::Diagnostic], file: &str, source: &str) {
    for diag in diagnostics {
        let severity = match diag.severity {
            fwgsl_diagnostics::Severity::Error => "error",
            fwgsl_diagnostics::Severity::Warning => "warning",
            fwgsl_diagnostics::Severity::Info => "info",
            fwgsl_diagnostics::Severity::Hint => "hint",
        };

        // Find line/col from first label span
        let location = if let Some(label) = diag.labels.first() {
            let (line, col) = offset_to_line_col(source, label.span.start as usize);
            format!("{}:{}:{}", file, line, col)
        } else {
            file.to_string()
        };

        eprintln!("{}: {}: {}", location, severity, diag.message);

        if let Some(ref help) = diag.help {
            eprintln!("  help: {}", help);
        }
    }
}

fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}
