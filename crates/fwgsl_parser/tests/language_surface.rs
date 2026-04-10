use fwgsl_parser::parser::{BindingAddressSpace, Decl, Expr, Parser, Type};

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
fn parse_index_expr() {
    let mut p = Parser::new("main m = m[0][1]");
    let program = p.parse_program();
    let Decl::FunDecl { body, .. } = &program.decls[0] else {
        panic!("fun")
    };
    assert!(matches!(body, Expr::Index(_, _, _)));
}
