//! Dead-code elimination via reachability analysis.
//!
//! Starting from entry points, walk the call graph to find all reachable
//! functions, structs, globals, and constants. Then filter the MIR program
//! to only include reachable declarations.

use std::collections::HashSet;

use crate::*;

/// The set of declarations reachable from entry points.
#[derive(Debug, Default)]
pub struct ReachableSet {
    pub functions: HashSet<String>,
    pub structs: HashSet<String>,
    pub globals: HashSet<String>,
    pub constants: HashSet<String>,
}

/// Compute which declarations are reachable from entry points.
pub fn compute_reachable(program: &MirProgram) -> ReachableSet {
    let mut reachable = ReachableSet::default();

    // Seed: walk all entry points
    for ep in &program.entry_points {
        walk_params(&ep.params, &mut reachable);
        walk_type(&ep.return_ty, &mut reachable);
        walk_stmts(&ep.body, &mut reachable);
        if let Some(expr) = &ep.return_expr {
            walk_expr(expr, &mut reachable);
        }
    }

    // Transitively resolve: functions can call other functions
    let mut worklist: Vec<String> = reachable.functions.iter().cloned().collect();
    let mut visited: HashSet<String> = reachable.functions.clone();

    while let Some(name) = worklist.pop() {
        if let Some(func) = program.functions.iter().find(|f| f.name == name) {
            walk_params(&func.params, &mut reachable);
            walk_type(&func.return_ty, &mut reachable);
            walk_stmts(&func.body, &mut reachable);
            if let Some(expr) = &func.return_expr {
                walk_expr(expr, &mut reachable);
            }
            // Check for newly discovered functions
            for new_fn in reachable.functions.difference(&visited).cloned().collect::<Vec<_>>() {
                visited.insert(new_fn.clone());
                worklist.push(new_fn);
            }
        }
    }

    // Constants can reference other things too
    let reachable_consts: Vec<String> = reachable.constants.iter().cloned().collect();
    for name in reachable_consts {
        if let Some(c) = program.constants.iter().find(|c| c.name == name) {
            walk_type(&c.ty, &mut reachable);
            walk_expr(&c.value, &mut reachable);
        }
    }

    // Transitively resolve struct dependencies
    let struct_names: Vec<String> = reachable.structs.iter().cloned().collect();
    for name in struct_names {
        mark_struct_deps(&name, &program.structs, &mut reachable);
    }

    // Also mark structs from globals
    let global_names: Vec<String> = reachable.globals.iter().cloned().collect();
    for name in global_names {
        if let Some(g) = program.globals.iter().find(|g| g.name == name) {
            walk_type(&g.ty, &mut reachable);
        }
    }

    // One more pass on struct deps after globals may have added new structs
    let struct_names: Vec<String> = reachable.structs.iter().cloned().collect();
    for name in struct_names {
        mark_struct_deps(&name, &program.structs, &mut reachable);
    }

    reachable
}

/// Filter a MIR program to only include reachable declarations.
pub fn filter_reachable(program: &MirProgram, reachable: &ReachableSet) -> MirProgram {
    MirProgram {
        structs: program
            .structs
            .iter()
            .filter(|s| reachable.structs.contains(&s.name))
            .cloned()
            .collect(),
        globals: program
            .globals
            .iter()
            .filter(|g| reachable.globals.contains(&g.name))
            .cloned()
            .collect(),
        functions: program
            .functions
            .iter()
            .filter(|f| reachable.functions.contains(&f.name))
            .cloned()
            .collect(),
        entry_points: program.entry_points.clone(),
        constants: program
            .constants
            .iter()
            .filter(|c| reachable.constants.contains(&c.name))
            .cloned()
            .collect(),
    }
}

/// Compute reachable declarations in library mode (no entry points).
///
/// Seeds reachability from ALL functions and ALL constants, keeping all
/// functions and constants but only structs/globals that they actually reference.
pub fn compute_reachable_library(program: &MirProgram) -> ReachableSet {
    let mut reachable = ReachableSet::default();

    // Seed: walk all functions
    for func in &program.functions {
        reachable.functions.insert(func.name.clone());
        walk_params(&func.params, &mut reachable);
        walk_type(&func.return_ty, &mut reachable);
        walk_stmts(&func.body, &mut reachable);
        if let Some(expr) = &func.return_expr {
            walk_expr(expr, &mut reachable);
        }
    }

    // Seed: keep all constants in library mode (they may have been promoted
    // from zero-param functions and are part of the module's public API).
    for c in &program.constants {
        reachable.constants.insert(c.name.clone());
        walk_type(&c.ty, &mut reachable);
        walk_expr(&c.value, &mut reachable);
    }

    // Transitively resolve struct dependencies
    let struct_names: Vec<String> = reachable.structs.iter().cloned().collect();
    for name in struct_names {
        mark_struct_deps(&name, &program.structs, &mut reachable);
    }

    // Also mark structs from globals
    let global_names: Vec<String> = reachable.globals.iter().cloned().collect();
    for name in global_names {
        if let Some(g) = program.globals.iter().find(|g| g.name == name) {
            walk_type(&g.ty, &mut reachable);
        }
    }

    // One more pass on struct deps after globals may have added new structs
    let struct_names: Vec<String> = reachable.structs.iter().cloned().collect();
    for name in struct_names {
        mark_struct_deps(&name, &program.structs, &mut reachable);
    }

    reachable
}

/// Filter a MIR program in library mode: keep all functions and constants,
/// but only reachable structs/globals.
pub fn filter_reachable_library(program: &MirProgram, reachable: &ReachableSet) -> MirProgram {
    MirProgram {
        structs: program
            .structs
            .iter()
            .filter(|s| reachable.structs.contains(&s.name))
            .cloned()
            .collect(),
        globals: program
            .globals
            .iter()
            .filter(|g| reachable.globals.contains(&g.name))
            .cloned()
            .collect(),
        functions: program.functions.clone(), // keep all functions in library mode
        entry_points: program.entry_points.clone(),
        constants: program.constants.clone(), // keep all constants in library mode
    }
}

/// Eliminate unreachable declarations from a MIR program.
///
/// If there are no entry points (library mode), all top-level functions are
/// kept but only structs/globals/constants reachable from those functions
/// survive.  This prevents unused prelude ADTs with unresolved type variables
/// from leaking into the output.
pub fn eliminate_dead_code(program: &MirProgram) -> MirProgram {
    if program.entry_points.is_empty() {
        let reachable = compute_reachable_library(program);
        return filter_reachable_library(program, &reachable);
    }
    let reachable = compute_reachable(program);
    filter_reachable(program, &reachable)
}

// ── Walkers ──────────────────────────────────────────────────────────────

fn walk_params(params: &[MirParam], reachable: &mut ReachableSet) {
    for p in params {
        walk_type(&p.ty, reachable);
    }
}

fn walk_type(ty: &MirType, reachable: &mut ReachableSet) {
    match ty {
        MirType::Struct(name) => {
            reachable.structs.insert(name.clone());
        }
        MirType::Vec(_, inner) | MirType::Array(inner, _) | MirType::RuntimeArray(inner) => walk_type(inner, reachable),
        MirType::Mat(_, _, inner) => walk_type(inner, reachable),
        _ => {}
    }
}

fn walk_stmts(stmts: &[MirStmt], reachable: &mut ReachableSet) {
    for stmt in stmts {
        walk_stmt(stmt, reachable);
    }
}

fn walk_stmt(stmt: &MirStmt, reachable: &mut ReachableSet) {
    match stmt {
        MirStmt::Let(_, ty, expr) | MirStmt::Var(_, ty, expr) => {
            walk_type(ty, reachable);
            walk_expr(expr, reachable);
        }
        MirStmt::Assign(_, expr) => walk_expr(expr, reachable),
        MirStmt::IndexAssign(base, index, val) => {
            walk_expr(base, reachable);
            walk_expr(index, reachable);
            walk_expr(val, reachable);
        }
        MirStmt::If(cond, then_stmts, else_stmts) => {
            walk_expr(cond, reachable);
            walk_stmts(then_stmts, reachable);
            walk_stmts(else_stmts, reachable);
        }
        MirStmt::Return(expr) => walk_expr(expr, reachable),
        MirStmt::Block(stmts) => walk_stmts(stmts, reachable),
        MirStmt::Switch(expr, cases, default) => {
            walk_expr(expr, reachable);
            for case in cases {
                walk_stmts(&case.body, reachable);
            }
            walk_stmts(default, reachable);
        }
        MirStmt::Loop(stmts) => walk_stmts(stmts, reachable),
        MirStmt::Break | MirStmt::Continue => {}
    }
}

fn walk_expr(expr: &MirExpr, reachable: &mut ReachableSet) {
    match expr {
        MirExpr::Lit(_) => {}
        MirExpr::Var(name, ty) => {
            // Globals are referenced by name via Var
            reachable.globals.insert(name.clone());
            // Also could be a constant
            reachable.constants.insert(name.clone());
            walk_type(ty, reachable);
        }
        MirExpr::BinOp(_, lhs, rhs, ty) => {
            walk_expr(lhs, reachable);
            walk_expr(rhs, reachable);
            walk_type(ty, reachable);
        }
        MirExpr::UnaryOp(_, operand, ty) => {
            walk_expr(operand, reachable);
            walk_type(ty, reachable);
        }
        MirExpr::Call(name, args, ty) => {
            reachable.functions.insert(name.clone());
            for arg in args {
                walk_expr(arg, reachable);
            }
            walk_type(ty, reachable);
        }
        MirExpr::ConstructStruct(name, fields) => {
            reachable.structs.insert(name.clone());
            for field in fields {
                walk_expr(field, reachable);
            }
        }
        MirExpr::FieldAccess(base, _, ty) => {
            walk_expr(base, reachable);
            walk_type(ty, reachable);
        }
        MirExpr::Index(base, index, ty) => {
            walk_expr(base, reachable);
            walk_expr(index, reachable);
            walk_type(ty, reachable);
        }
        MirExpr::Cast(inner, ty) => {
            walk_expr(inner, reachable);
            walk_type(ty, reachable);
        }
    }
}

/// Transitively mark struct dependencies (structs containing other structs).
fn mark_struct_deps(name: &str, structs: &[MirStruct], reachable: &mut ReachableSet) {
    if let Some(s) = structs.iter().find(|s| s.name == name) {
        for field in &s.fields {
            walk_type_for_struct_deps(&field.ty, structs, reachable);
        }
    }
}

fn walk_type_for_struct_deps(ty: &MirType, structs: &[MirStruct], reachable: &mut ReachableSet) {
    match ty {
        MirType::Struct(dep) => {
            if reachable.structs.insert(dep.clone()) {
                mark_struct_deps(dep, structs, reachable);
            }
        }
        MirType::Vec(_, inner) | MirType::Array(inner, _) | MirType::RuntimeArray(inner) => {
            walk_type_for_struct_deps(inner, structs, reachable);
        }
        MirType::Mat(_, _, inner) => {
            walk_type_for_struct_deps(inner, structs, reachable);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_fn(name: &str, calls: &[&str]) -> MirFunction {
        let body: Vec<MirStmt> = calls
            .iter()
            .map(|c| {
                MirStmt::Let(
                    format!("_tmp_{}", c),
                    MirType::I32,
                    MirExpr::Call(c.to_string(), vec![], MirType::I32),
                )
            })
            .collect();
        MirFunction {
            name: name.to_string(),
            params: vec![],
            return_ty: MirType::I32,
            body,
            return_expr: Some(MirExpr::Lit(MirLit::I32(0))),
            comments: vec![],
        }
    }

    fn make_entry_point(name: &str, calls: &[&str]) -> MirEntryPoint {
        let body: Vec<MirStmt> = calls
            .iter()
            .map(|c| {
                MirStmt::Let(
                    format!("_tmp_{}", c),
                    MirType::I32,
                    MirExpr::Call(c.to_string(), vec![], MirType::I32),
                )
            })
            .collect();
        MirEntryPoint {
            name: name.to_string(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([64, 1, 1]),
            params: vec![],
            return_ty: MirType::Unit,
            body,
            return_expr: None,
            comments: vec![],
        }
    }

    #[test]
    fn unused_function_is_eliminated() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
            functions: vec![
                make_simple_fn("used", &[]),
                make_simple_fn("unused", &[]),
            ],
            entry_points: vec![make_entry_point("main", &["used"])],
            constants: vec![],
        };

        let result = eliminate_dead_code(&program);
        assert_eq!(result.functions.len(), 1);
        assert_eq!(result.functions[0].name, "used");
    }

    #[test]
    fn transitive_call_keeps_both() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![],
            functions: vec![
                make_simple_fn("a", &["b"]),
                make_simple_fn("b", &[]),
                make_simple_fn("c", &[]),
            ],
            entry_points: vec![make_entry_point("main", &["a"])],
            constants: vec![],
        };

        let result = eliminate_dead_code(&program);
        let names: HashSet<&str> = result.functions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains("a"));
        assert!(names.contains("b"));
        assert!(!names.contains("c"));
    }

    #[test]
    fn unused_struct_is_eliminated() {
        let program = MirProgram {
            structs: vec![
                MirStruct {
                    name: "Used".to_string(),
                    fields: vec![MirField {
                        name: "x".to_string(),
                        ty: MirType::F32,
                        attributes: vec![],
                    }],
                },
                MirStruct {
                    name: "Unused".to_string(),
                    fields: vec![MirField {
                        name: "y".to_string(),
                        ty: MirType::I32,
                        attributes: vec![],
                    }],
                },
            ],
            globals: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
                params: vec![],
                return_ty: MirType::Unit,
                body: vec![MirStmt::Let(
                    "s".to_string(),
                    MirType::Struct("Used".to_string()),
                    MirExpr::ConstructStruct("Used".to_string(), vec![MirExpr::Lit(MirLit::F32(1.0))]),
                )],
                return_expr: None,
                comments: vec![],
            }],
            constants: vec![],
        };

        let result = eliminate_dead_code(&program);
        assert_eq!(result.structs.len(), 1);
        assert_eq!(result.structs[0].name, "Used");
    }

    #[test]
    fn unused_global_is_eliminated() {
        let program = MirProgram {
            structs: vec![],
            globals: vec![
                MirGlobal {
                    name: "used_buf".to_string(),
                    address_space: AddressSpace::StorageReadWrite,
                    ty: MirType::Array(Box::new(MirType::F32), 64),
                    group: 0,
                    binding: 0,
                },
                MirGlobal {
                    name: "unused_buf".to_string(),
                    address_space: AddressSpace::Uniform,
                    ty: MirType::F32,
                    group: 0,
                    binding: 1,
                },
            ],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([64, 1, 1]),
                params: vec![],
                return_ty: MirType::Unit,
                body: vec![MirStmt::IndexAssign(
                    MirExpr::Var("used_buf".to_string(), MirType::Array(Box::new(MirType::F32), 64)),
                    MirExpr::Lit(MirLit::I32(0)),
                    MirExpr::Lit(MirLit::F32(1.0)),
                )],
                return_expr: None,
                comments: vec![],
            }],
            constants: vec![],
        };

        let result = eliminate_dead_code(&program);
        assert_eq!(result.globals.len(), 1);
        assert_eq!(result.globals[0].name, "used_buf");
    }

    #[test]
    fn struct_used_via_global_resource_is_kept() {
        let program = MirProgram {
            structs: vec![MirStruct {
                name: "Particle".to_string(),
                fields: vec![MirField {
                    name: "pos".to_string(),
                    ty: MirType::Vec(3, Box::new(MirType::F32)),
                    attributes: vec![],
                }],
            }],
            globals: vec![MirGlobal {
                name: "particles".to_string(),
                address_space: AddressSpace::StorageReadWrite,
                ty: MirType::Array(Box::new(MirType::Struct("Particle".to_string())), 256),
                group: 0,
                binding: 0,
            }],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([64, 1, 1]),
                params: vec![],
                return_ty: MirType::Unit,
                body: vec![MirStmt::Let(
                    "p".to_string(),
                    MirType::Struct("Particle".to_string()),
                    MirExpr::Index(
                        Box::new(MirExpr::Var(
                            "particles".to_string(),
                            MirType::Array(Box::new(MirType::Struct("Particle".to_string())), 256),
                        )),
                        Box::new(MirExpr::Lit(MirLit::I32(0))),
                        MirType::Struct("Particle".to_string()),
                    ),
                )],
                return_expr: None,
                comments: vec![],
            }],
            constants: vec![],
        };

        let result = eliminate_dead_code(&program);
        assert_eq!(result.structs.len(), 1);
        assert_eq!(result.structs[0].name, "Particle");
        assert_eq!(result.globals.len(), 1);
    }

    #[test]
    fn no_entry_points_keeps_functions_eliminates_unused_structs() {
        let program = MirProgram {
            structs: vec![MirStruct {
                name: "Foo".to_string(),
                fields: vec![],
            }],
            globals: vec![],
            functions: vec![make_simple_fn("helper", &[])],
            entry_points: vec![],
            constants: vec![],
        };

        let result = eliminate_dead_code(&program);
        assert_eq!(result.functions.len(), 1);
        // Foo is not referenced by any function, so it gets eliminated
        assert_eq!(result.structs.len(), 0);
    }

    #[test]
    fn no_entry_points_keeps_used_structs() {
        let program = MirProgram {
            structs: vec![
                MirStruct {
                    name: "Used".to_string(),
                    fields: vec![MirField {
                        name: "x".to_string(),
                        ty: MirType::F32,
                        attributes: vec![],
                    }],
                },
                MirStruct {
                    name: "Unused".to_string(),
                    fields: vec![],
                },
            ],
            globals: vec![],
            functions: vec![MirFunction {
                name: "helper".to_string(),
                params: vec![MirParam {
                    name: "s".to_string(),
                    ty: MirType::Struct("Used".to_string()),
                }],
                return_ty: MirType::Struct("Used".to_string()),
                body: vec![],
                return_expr: Some(MirExpr::Var("s".to_string(), MirType::Struct("Used".to_string()))),
                comments: vec![],
            }],
            entry_points: vec![],
            constants: vec![],
        };

        let result = eliminate_dead_code(&program);
        assert_eq!(result.functions.len(), 1);
        assert_eq!(result.structs.len(), 1);
        assert_eq!(result.structs[0].name, "Used");
    }

    #[test]
    fn nested_struct_deps_are_kept() {
        let program = MirProgram {
            structs: vec![
                MirStruct {
                    name: "Inner".to_string(),
                    fields: vec![MirField {
                        name: "v".to_string(),
                        ty: MirType::F32,
                        attributes: vec![],
                    }],
                },
                MirStruct {
                    name: "Outer".to_string(),
                    fields: vec![MirField {
                        name: "inner".to_string(),
                        ty: MirType::Struct("Inner".to_string()),
                        attributes: vec![],
                    }],
                },
                MirStruct {
                    name: "Unrelated".to_string(),
                    fields: vec![],
                },
            ],
            globals: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
                params: vec![MirParam {
                    name: "o".to_string(),
                    ty: MirType::Struct("Outer".to_string()),
                }],
                return_ty: MirType::Unit,
                body: vec![],
                return_expr: None,
                comments: vec![],
            }],
            constants: vec![],
        };

        let result = eliminate_dead_code(&program);
        let names: HashSet<&str> = result.structs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains("Inner"));
        assert!(names.contains("Outer"));
        assert!(!names.contains("Unrelated"));
    }
}
