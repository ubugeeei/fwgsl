///|
/// Semantic analysis for fwgsl.
///
/// Performs name resolution and type inference over the parser's AST.
/// Collects data type definitions, constructor info, type signatures, and
/// infers types for function bodies using Algorithm W (HM inference).
///
/// This file ports the structural pieces of `crates/fwgsl_semantic/src/lib.rs`:
/// the `SemanticAnalyzer` struct, the `DataTypeInfo` record, builtin data
/// type registration (`Option`, `Result`, `Pair`), and the arithmetic /
/// comparison / boolean operator schemes. The full builtin prelude and the
/// inference engine over AST expressions remain as a follow-up; the Rust
/// crate fills ~1700 lines of similar registration code.

///|
pub(all) struct DataTypeInfo {
  name : String
  type_params : Array[String]
  constructors : Array[String]
}

///|
pub struct SemanticAnalyzer {
  env : @typechecker.TypeEnv
  engine : @typechecker.InferEngine
  constructors : Map[String, @typechecker.ConstructorInfo]
  data_types : Map[String, DataTypeInfo]
}

///|
pub fn SemanticAnalyzer::new() -> SemanticAnalyzer {
  let sa = SemanticAnalyzer::{
    env: @typechecker.TypeEnv::new(),
    engine: @typechecker.InferEngine::new(),
    constructors: Map::new(),
    data_types: Map::new(),
  }
  sa.add_builtins()
  sa
}

///|
pub fn SemanticAnalyzer::env(self : SemanticAnalyzer) -> @typechecker.TypeEnv {
  self.env
}

///|
pub fn SemanticAnalyzer::engine(self : SemanticAnalyzer) -> @typechecker.InferEngine {
  self.engine
}

///|
pub fn SemanticAnalyzer::constructors(
  self : SemanticAnalyzer
) -> Map[String, @typechecker.ConstructorInfo] {
  self.constructors
}

///|
pub fn SemanticAnalyzer::data_types(
  self : SemanticAnalyzer
) -> Map[String, DataTypeInfo] {
  self.data_types
}

///|
fn fresh_var_id(engine : @typechecker.InferEngine) -> UInt {
  match engine.fresh_var() {
    Var(v) => v
    _ => 0
  }
}

///|
fn SemanticAnalyzer::add_builtins(self : SemanticAnalyzer) -> Unit {
  self.add_builtin_data_types()
  self.add_builtin_operator_schemes()
}

///|
fn SemanticAnalyzer::add_builtin_data_types(self : SemanticAnalyzer) -> Unit {
  self.register_builtin_data_type(
    "Option",
    ["a"],
    [
      ("Some", @typechecker.ConstructorFields::Positional([@typechecker.Ty::Var(0)])),
      ("None", @typechecker.ConstructorFields::Empty),
    ],
  )
  self.register_builtin_data_type(
    "Result",
    ["a"],
    [
      ("Ok", @typechecker.ConstructorFields::Positional([@typechecker.Ty::Var(0)])),
      (
        "Err",
        @typechecker.ConstructorFields::Positional([@typechecker.Ty::Con("String")]),
      ),
    ],
  )
  self.register_builtin_data_type(
    "Pair",
    ["a", "b"],
    [
      (
        "Pair",
        @typechecker.ConstructorFields::Positional(
          [@typechecker.Ty::Var(0), @typechecker.Ty::Var(1)],
        ),
      ),
    ],
  )
}

///|
fn SemanticAnalyzer::register_builtin_data_type(
  self : SemanticAnalyzer,
  name : String,
  type_params : Array[String],
  ctors : Array[(String, @typechecker.ConstructorFields)]
) -> Unit {
  let constructor_names : Array[String] = Array::new()
  let scheme_vars : Array[UInt] = Array::new()
  let mut i : UInt = 0
  while i.reinterpret_as_int() < type_params.length() {
    scheme_vars.push(i)
    i = i + 1
  }
  let result_ty = build_data_ty(name, scheme_vars)
  let mut tag : UInt = 0
  for con_pair in ctors {
    let (con_name, fields) = con_pair
    constructor_names.push(con_name)
    self.constructors[con_name] = @typechecker.ConstructorInfo::{
      type_name: name,
      tag,
      scheme_vars,
      fields,
      result_ty,
    }
    tag = tag + 1
  }
  self.data_types[name] = DataTypeInfo::{
    name,
    type_params,
    constructors: constructor_names,
  }
}

///|
fn build_data_ty(name : String, scheme_vars : Array[UInt]) -> @typechecker.Ty {
  let mut ty : @typechecker.Ty = @typechecker.Ty::Con(name)
  for v in scheme_vars {
    ty = @typechecker.Ty::App(ty, @typechecker.Ty::Var(v))
  }
  ty
}

///|
fn SemanticAnalyzer::add_builtin_operator_schemes(
  self : SemanticAnalyzer
) -> Unit {
  // Arithmetic: a -> a -> a
  let a1 = fresh_var_id(self.engine)
  let numeric_binop = @typechecker.Scheme::poly(
    [a1],
    @typechecker.Ty::arrow(
      @typechecker.Ty::Var(a1),
      @typechecker.Ty::arrow(@typechecker.Ty::Var(a1), @typechecker.Ty::Var(a1)),
    ),
  )
  for op in ["+", "-", "*", "/", "%"] {
    self.env.insert(op, numeric_binop)
  }
  // Comparison: a -> a -> Bool
  let a2 = fresh_var_id(self.engine)
  let cmp_binop = @typechecker.Scheme::poly(
    [a2],
    @typechecker.Ty::arrow(
      @typechecker.Ty::Var(a2),
      @typechecker.Ty::arrow(@typechecker.Ty::Var(a2), @typechecker.Ty::bool()),
    ),
  )
  for op in ["==", "/=", "<", ">", "<=", ">="] {
    self.env.insert(op, cmp_binop)
  }
  // Boolean: Bool -> Bool -> Bool
  let bool_binop = @typechecker.Scheme::mono(
    @typechecker.Ty::arrow(
      @typechecker.Ty::bool(),
      @typechecker.Ty::arrow(@typechecker.Ty::bool(), @typechecker.Ty::bool()),
    ),
  )
  self.env.insert("&&", bool_binop)
  self.env.insert("||", bool_binop)
}

///|
/// Collect declarations from a `Program` AST: register data-type definitions
/// and absorb explicit type signatures into the environment.
///
/// The Rust crate also runs full HM inference here; that is left as the
/// next milestone in #4.
pub fn SemanticAnalyzer::analyze(
  self : SemanticAnalyzer,
  prog : @ast.Program
) -> Unit {
  for d in prog.decls {
    match d {
      DataDecl(name, type_params, constructors, _) =>
        self.register_data_decl(name, type_params, constructors)
      TypeSig(name, ty, _) => self.register_type_sig(name, ty)
      _ => ()
    }
  }
}

///|
fn SemanticAnalyzer::register_data_decl(
  self : SemanticAnalyzer,
  name : String,
  type_params : Array[String],
  constructors : Array[@ast.ConDecl]
) -> Unit {
  let scheme_vars : Array[UInt] = Array::new()
  let var_map : Map[String, UInt] = Map::new()
  for p in type_params {
    let id = fresh_var_id(self.engine)
    scheme_vars.push(id)
    var_map[p] = id
  }
  let result_ty = build_data_ty(name, scheme_vars)
  let constructor_names : Array[String] = Array::new()
  let mut tag : UInt = 0
  for con in constructors {
    constructor_names.push(con.name)
    let fields = ast_fields_to_ctor_fields(con.fields, var_map)
    self.constructors[con.name] = @typechecker.ConstructorInfo::{
      type_name: name,
      tag,
      scheme_vars,
      fields,
      result_ty,
    }
    tag = tag + 1
  }
  self.data_types[name] = DataTypeInfo::{
    name,
    type_params,
    constructors: constructor_names,
  }
}

///|
fn ast_fields_to_ctor_fields(
  fields : @ast.ConFields,
  var_map : Map[String, UInt]
) -> @typechecker.ConstructorFields {
  match fields {
    Positional(tys) => {
      let out : Array[@typechecker.Ty] = Array::new()
      for ty in tys {
        out.push(ast_ty_to_ty(ty, var_map))
      }
      @typechecker.ConstructorFields::Positional(out)
    }
    RecordFields(rs) => {
      let out : Array[(String, @typechecker.Ty)] = Array::new()
      for r in rs {
        let (n, ty) = r
        out.push((n, ast_ty_to_ty(ty, var_map)))
      }
      @typechecker.ConstructorFields::RecordFields(out)
    }
    Empty => @typechecker.ConstructorFields::Empty
  }
}

///|
/// Translate an `@ast.Type` into a `@typechecker.Ty`. Type variables are
/// resolved through `var_map` so multiple references within the same
/// constructor share the same fresh `TyVarId`.
fn ast_ty_to_ty(
  ty : @ast.Type,
  var_map : Map[String, UInt]
) -> @typechecker.Ty {
  match ty {
    Con(name, _) => @typechecker.Ty::Con(name)
    Var(name, _) =>
      match var_map.get(name) {
        Some(v) => @typechecker.Ty::Var(v)
        None => @typechecker.Ty::Con(name)
      }
    Nat(n, _) => @typechecker.Ty::Nat(n)
    App(f, arg, _) =>
      @typechecker.Ty::App(ast_ty_to_ty(f, var_map), ast_ty_to_ty(arg, var_map))
    Arrow(a, b, _) =>
      @typechecker.Ty::arrow(
        ast_ty_to_ty(a, var_map),
        ast_ty_to_ty(b, var_map),
      )
    Paren(inner, _) => ast_ty_to_ty(inner, var_map)
    Tuple(elems, _) => {
      // Tuples are not modelled in the Ty system yet; encode as a chain
      // of `Pair` for parity with the typechecker helpers.
      let mut acc : @typechecker.Ty = match elems.length() {
        0 => @typechecker.Ty::unit()
        _ => ast_ty_to_ty(elems[elems.length() - 1], var_map)
      }
      let mut i = elems.length() - 2
      while i >= 0 {
        acc = @typechecker.pair_ty(ast_ty_to_ty(elems[i], var_map), acc)
        i = i - 1
      }
      acc
    }
    Unit(_) => @typechecker.Ty::unit()
  }
}

///|
fn SemanticAnalyzer::register_type_sig(
  self : SemanticAnalyzer,
  name : String,
  ty : @ast.Type
) -> Unit {
  let var_map : Map[String, UInt] = Map::new()
  let ty = ast_ty_to_ty(ty, var_map)
  // Generalize over any free variables we introduced.
  let scheme = self.engine.generalize(self.env, ty)
  self.env.insert(name, scheme)
}
