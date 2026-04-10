use shadml_parser::parser::{BindingAddressSpace, Decl, Expr, Parser, Type};
use shadml_parser::{evaluate_features, FeatureSet};

#[test]
fn pipeline_inserts_lhs_as_first_arg() {
    let mut p = Parser::new("main x = x |> clamp 0 1");
    let program = p.parse_program();
    let Decl::FunDecl { body, .. } = &program.decls[0] else {
        panic!("expected fun");
    };
    match body {
        Expr::App(fab, one, _) => {
            assert!(matches!(**one, Expr::Lit(_, _)));
            match &**fab {
                Expr::App(fx, zero, _) => {
                    assert!(matches!(**zero, Expr::Lit(_, _)));
                    match &**fx {
                        Expr::App(clamp, x, _) => {
                            assert!(matches!(**clamp, Expr::Var(ref n, _) if n == "clamp"));
                            assert!(matches!(**x, Expr::Var(ref n, _) if n == "x"));
                        }
                        _ => panic!("expected clamp x"),
                    }
                }
                _ => panic!("expected (clamp x) 0"),
            }
        }
        _ => panic!("unexpected body"),
    }
}

#[test]
fn parse_angle_type_arguments() {
    let mut p = Parser::new("alias Float2 = Vec<2, F32>");
    let program = p.parse_program();
    let Decl::TypeAlias { ty, .. } = &program.decls[0] else {
        panic!("alias")
    };
    match ty {
        Type::App(_, _, _) => {}
        _ => panic!("expected applied type"),
    }
}

#[test]
fn parse_binding_decl_uniform() {
    let mut p = Parser::new("@group(0) @binding(1) uniform frame : FrameData");
    let program = p.parse_program();
    let Decl::BindingDecl {
        name,
        address_space,
        group,
        binding,
        ..
    } = &program.decls[0]
    else {
        panic!("binding")
    };
    assert_eq!(name, "frame");
    assert_eq!(*address_space, BindingAddressSpace::Uniform);
    assert_eq!(*group, 0);
    assert_eq!(*binding, 1);
}

#[test]
fn parse_binding_decl_storage_default() {
    let mut p = Parser::new("@group(1) @binding(0) storage output : Array<F32, 1024>");
    let program = p.parse_program();
    let Decl::BindingDecl {
        name,
        address_space,
        group,
        binding,
        ..
    } = &program.decls[0]
    else {
        panic!("binding")
    };
    assert_eq!(name, "output");
    assert_eq!(*address_space, BindingAddressSpace::StorageRead);
    assert_eq!(*group, 1);
    assert_eq!(*binding, 0);
}

#[test]
fn parse_binding_decl_storage_read_write() {
    let mut p =
        Parser::new("@group(0) @binding(3) storage(read_write) output : Array<Vec<4, F32>, 64>");
    let program = p.parse_program();
    let Decl::BindingDecl {
        name,
        address_space,
        group,
        binding,
        ..
    } = &program.decls[0]
    else {
        panic!("binding")
    };
    assert_eq!(name, "output");
    assert_eq!(*address_space, BindingAddressSpace::StorageReadWrite);
    assert_eq!(*group, 0);
    assert_eq!(*binding, 3);
}

#[test]
fn parse_binding_group_block() {
    let src = "\
@group(0)
  @binding(0) uniform             frame  : FrameData
  @binding(1) storage(read_write) output : Array<F32, 64>
";
    let mut p = Parser::new(src);
    let program = p.parse_program();
    assert_eq!(program.decls.len(), 2, "group block should produce 2 bindings");

    let Decl::BindingDecl { name, address_space, group, binding, .. } = &program.decls[0]
    else { panic!("binding 0") };
    assert_eq!(name, "frame");
    assert_eq!(*address_space, BindingAddressSpace::Uniform);
    assert_eq!(*group, 0);
    assert_eq!(*binding, 0);

    let Decl::BindingDecl { name, address_space, group, binding, .. } = &program.decls[1]
    else { panic!("binding 1") };
    assert_eq!(name, "output");
    assert_eq!(*address_space, BindingAddressSpace::StorageReadWrite);
    assert_eq!(*group, 0);
    assert_eq!(*binding, 1);
}

#[test]
fn parse_binding_inside_when_block() {
    let src = "\
when cfg.debug
  @group(0) @binding(5) uniform debugBuf : F32
f x = x
";
    let mut p = Parser::new(src);
    let mut program = p.parse_program();

    // With debug enabled: binding should be present
    let features = FeatureSet::from_flags(&["debug".to_string()]);
    let mut prog_on = program.clone();
    evaluate_features(&mut prog_on, &features);
    assert_eq!(
        prog_on.decls.iter().filter(|d| matches!(d, Decl::BindingDecl { .. })).count(),
        1,
        "binding should be present when debug is enabled"
    );

    // Without debug: binding should be pruned
    let features = FeatureSet::new();
    evaluate_features(&mut program, &features);
    assert_eq!(
        program.decls.iter().filter(|d| matches!(d, Decl::BindingDecl { .. })).count(),
        0,
        "binding should be absent when debug is disabled"
    );
}

#[test]
fn parse_group_block_inside_when_block() {
    let src = "\
when cfg.debug
  @group(2)
    @binding(0) uniform debugA : F32
    @binding(1) uniform debugB : F32
f x = x
";
    let mut p = Parser::new(src);
    let mut program = p.parse_program();

    // With debug enabled: both bindings present
    let features = FeatureSet::from_flags(&["debug".to_string()]);
    let mut prog_on = program.clone();
    evaluate_features(&mut prog_on, &features);
    let bindings: Vec<_> = prog_on.decls.iter().filter(|d| matches!(d, Decl::BindingDecl { .. })).collect();
    assert_eq!(bindings.len(), 2, "group block should produce 2 bindings inside when block");

    // Without debug: no bindings
    let features = FeatureSet::new();
    evaluate_features(&mut program, &features);
    assert_eq!(
        program.decls.iter().filter(|d| matches!(d, Decl::BindingDecl { .. })).count(),
        0,
        "bindings should be absent when debug is disabled"
    );
}

#[test]
fn parse_index_expr() {
    let mut p = Parser::new("main m = m[0][1]");
    let program = p.parse_program();
    let Decl::FunDecl { body, .. } = &program.decls[0] else {
        panic!("fun")
    };
    assert!(matches!(body, Expr::Index(_, _, _)));
}
