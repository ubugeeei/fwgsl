///|
/// Semantic analysis for fwgsl.
///
/// Performs name resolution and type inference over the parser's AST.
/// Collects data type definitions, constructor info, type signatures,
/// and infers types for function bodies using Algorithm W (HM inference).

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
pub fn SemanticAnalyzer::has_errors(self : SemanticAnalyzer) -> Bool {
  self.engine.diagnostics().has_errors()
}

///|
pub fn SemanticAnalyzer::diagnostics(
  self : SemanticAnalyzer
) -> @diagnostics.DiagnosticSink {
  self.engine.diagnostics()
}

///|
fn fresh_var_id(engine : @typechecker.InferEngine) -> UInt {
  match engine.fresh_var() {
    Var(v) => v
    _ => 0
  }
}

///|
fn builtin_span() -> @span.Span {
  @span.Span::new(0, 0)
}

// ---------------------------------------------------------------------------
// Builtin registration
// ---------------------------------------------------------------------------

///|
fn SemanticAnalyzer::add_builtins(self : SemanticAnalyzer) -> Unit {
  self.add_builtin_data_types()
  self.add_builtin_operator_schemes()
  self.add_builtin_value_schemes()
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
    // Build the constructor's function type from its fields, then insert as
    // a polymorphic scheme so callers can use it as a value.
    let con_ty = match fields {
      Empty => result_ty
      Positional(field_tys) => arrow_fold(field_tys, result_ty)
      RecordFields(record_fields) => {
        let tys : Array[@typechecker.Ty] = Array::new()
        for r in record_fields {
          let (_, ty) = r
          tys.push(ty)
        }
        arrow_fold(tys, result_ty)
      }
    }
    self.env.insert(con_name, @typechecker.Scheme::poly(scheme_vars, con_ty))
    tag = tag + 1
  }
  self.data_types[name] = DataTypeInfo::{
    name,
    type_params,
    constructors: constructor_names,
  }
}

///|
fn arrow_fold(
  fields : Array[@typechecker.Ty],
  result_ty : @typechecker.Ty
) -> @typechecker.Ty {
  let mut ty = result_ty
  let mut i = fields.length() - 1
  while i >= 0 {
    ty = @typechecker.Ty::arrow(fields[i], ty)
    i = i - 1
  }
  ty
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
fn SemanticAnalyzer::insert_builtin(
  self : SemanticAnalyzer,
  names : Array[String],
  scheme : @typechecker.Scheme
) -> Unit {
  for name in names {
    self.env.insert(name, scheme)
  }
}

///|
fn SemanticAnalyzer::add_builtin_value_schemes(
  self : SemanticAnalyzer
) -> Unit {
  // id : a -> a
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["id"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::Var(a)),
    ),
  )
  // const : a -> b -> a
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  self.insert_builtin(
    ["const"],
    @typechecker.Scheme::poly(
      [a, b],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.Ty::arrow(@typechecker.Ty::Var(b), @typechecker.Ty::Var(a)),
      ),
    ),
  )
  // flip : (a -> b -> c) -> b -> a -> c
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let c = fresh_var_id(self.engine)
  self.insert_builtin(
    ["flip"],
    @typechecker.Scheme::poly(
      [a, b, c],
      @typechecker.Ty::arrow(
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(a),
          @typechecker.Ty::arrow(@typechecker.Ty::Var(b), @typechecker.Ty::Var(c)),
        ),
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(b),
          @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::Var(c)),
        ),
      ),
    ),
  )
  // compose : (b -> c) -> (a -> b) -> (a -> c)
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let c = fresh_var_id(self.engine)
  self.insert_builtin(
    ["compose"],
    @typechecker.Scheme::poly(
      [a, b, c],
      @typechecker.Ty::arrow(
        @typechecker.Ty::arrow(@typechecker.Ty::Var(b), @typechecker.Ty::Var(c)),
        @typechecker.Ty::arrow(
          @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::Var(b)),
          @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::Var(c)),
        ),
      ),
    ),
  )
  // pure : a -> f a
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["pure"],
    @typechecker.Scheme::poly(
      [f, a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a)),
      ),
    ),
  )
  // bind : f a -> (a -> f b) -> f b
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let fa = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a))
  let fb = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(b))
  self.insert_builtin(
    ["bind"],
    @typechecker.Scheme::poly(
      [f, a, b],
      @typechecker.Ty::arrow(
        fa,
        @typechecker.Ty::arrow(
          @typechecker.Ty::arrow(@typechecker.Ty::Var(a), fb),
          fb,
        ),
      ),
    ),
  )
  // fmap / map / $map : (a -> b) -> f a -> f b
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let fa = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a))
  let fb = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(b))
  let map_scheme = @typechecker.Scheme::poly(
    [f, a, b],
    @typechecker.Ty::arrow(
      @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::Var(b)),
      @typechecker.Ty::arrow(fa, fb),
    ),
  )
  self.insert_builtin(["fmap", "map", "$map"], map_scheme)
  // filter / $filter : (a -> Bool) -> f a -> f a
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  let fa = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a))
  self.insert_builtin(
    ["filter", "$filter"],
    @typechecker.Scheme::poly(
      [f, a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::bool()),
        @typechecker.Ty::arrow(fa, fa),
      ),
    ),
  )
  // foldl / fold / $fold : (b -> a -> b) -> b -> f a -> b
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let fa = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a))
  self.insert_builtin(
    ["foldl", "fold", "$fold"],
    @typechecker.Scheme::poly(
      [f, a, b],
      @typechecker.Ty::arrow(
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(b),
          @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::Var(b)),
        ),
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(b),
          @typechecker.Ty::arrow(fa, @typechecker.Ty::Var(b)),
        ),
      ),
    ),
  )
  // foldr / $foldr : (a -> b -> b) -> b -> f a -> b
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let fa = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a))
  self.insert_builtin(
    ["foldr", "$foldr"],
    @typechecker.Scheme::poly(
      [f, a, b],
      @typechecker.Ty::arrow(
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(a),
          @typechecker.Ty::arrow(@typechecker.Ty::Var(b), @typechecker.Ty::Var(b)),
        ),
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(b),
          @typechecker.Ty::arrow(fa, @typechecker.Ty::Var(b)),
        ),
      ),
    ),
  )
  // flat / $flat : f (f a) -> f a
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  let fa = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a))
  let ffa = @typechecker.Ty::app(@typechecker.Ty::Var(f), fa)
  self.insert_builtin(
    ["flat", "$flat"],
    @typechecker.Scheme::poly([f, a], @typechecker.Ty::arrow(ffa, fa)),
  )
  // flatMap / $flatMap : (a -> f b) -> f a -> f b
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let fa = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a))
  let fb = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(b))
  self.insert_builtin(
    ["flatMap", "$flatMap"],
    @typechecker.Scheme::poly(
      [f, a, b],
      @typechecker.Ty::arrow(
        @typechecker.Ty::arrow(@typechecker.Ty::Var(a), fb),
        @typechecker.Ty::arrow(fa, fb),
      ),
    ),
  )
  // zip / $zip : f a -> f b -> f (Pair a b)
  let f = fresh_var_id(self.engine)
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let pair_ab = @typechecker.pair_ty(@typechecker.Ty::Var(a), @typechecker.Ty::Var(b))
  let fa = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(a))
  let fb_ = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::Var(b))
  self.insert_builtin(
    ["zip", "$zip"],
    @typechecker.Scheme::poly(
      [f, a, b],
      @typechecker.Ty::arrow(
        fa,
        @typechecker.Ty::arrow(
          fb_,
          @typechecker.Ty::app(@typechecker.Ty::Var(f), pair_ab),
        ),
      ),
    ),
  )
  // all / any : f Bool -> Bool
  let f = fresh_var_id(self.engine)
  let bools = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::bool())
  self.insert_builtin(
    ["all", "$all"],
    @typechecker.Scheme::poly([f], @typechecker.Ty::arrow(bools, @typechecker.Ty::bool())),
  )
  let f = fresh_var_id(self.engine)
  let bools = @typechecker.Ty::app(@typechecker.Ty::Var(f), @typechecker.Ty::bool())
  self.insert_builtin(
    ["any", "$any"],
    @typechecker.Scheme::poly([f], @typechecker.Ty::arrow(bools, @typechecker.Ty::bool())),
  )
  // fst / snd / swap on Pair
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let pair_ab = @typechecker.pair_ty(@typechecker.Ty::Var(a), @typechecker.Ty::Var(b))
  self.insert_builtin(
    ["fst", "$fst"],
    @typechecker.Scheme::poly(
      [a, b],
      @typechecker.Ty::arrow(pair_ab, @typechecker.Ty::Var(a)),
    ),
  )
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let pair_ab = @typechecker.pair_ty(@typechecker.Ty::Var(a), @typechecker.Ty::Var(b))
  self.insert_builtin(
    ["snd", "$snd"],
    @typechecker.Scheme::poly(
      [a, b],
      @typechecker.Ty::arrow(pair_ab, @typechecker.Ty::Var(b)),
    ),
  )
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  let pair_ab = @typechecker.pair_ty(@typechecker.Ty::Var(a), @typechecker.Ty::Var(b))
  let pair_ba = @typechecker.pair_ty(@typechecker.Ty::Var(b), @typechecker.Ty::Var(a))
  self.insert_builtin(
    ["swap", "$swap"],
    @typechecker.Scheme::poly([a, b], @typechecker.Ty::arrow(pair_ab, pair_ba)),
  )
  // unwrapOr : a -> Option a -> a
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["unwrapOr", "$unwrapOr"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.Ty::arrow(
          @typechecker.option_ty(@typechecker.Ty::Var(a)),
          @typechecker.Ty::Var(a),
        ),
      ),
    ),
  )
  // Unary numeric: $sin / $cos / $abs / $fract / $floor / $sign / $sqrt / negate
  let a = fresh_var_id(self.engine)
  let unary_numeric = @typechecker.Scheme::poly(
    [a],
    @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::Var(a)),
  )
  self.insert_builtin(
    [
      "$sin", "$cos", "$abs", "$fract", "$floor", "$sign", "$sqrt", "negate",
    ],
    unary_numeric,
  )
  // $normalize : Vec n a -> Vec n a
  let dim = fresh_var_id(self.engine)
  let scalar = fresh_var_id(self.engine)
  let vec_dim_scalar = @typechecker.Ty::app(
    @typechecker.Ty::app(@typechecker.Ty::Con("Vec"), @typechecker.Ty::Var(dim)),
    @typechecker.Ty::Var(scalar),
  )
  self.insert_builtin(
    ["$normalize"],
    @typechecker.Scheme::poly(
      [dim, scalar],
      @typechecker.Ty::arrow(vec_dim_scalar, vec_dim_scalar),
    ),
  )
  // Binary numeric: $max / $min / $step / $mod / $pow / $reflect / $atan
  let a = fresh_var_id(self.engine)
  let binary_numeric = @typechecker.Scheme::poly(
    [a],
    @typechecker.Ty::arrow(
      @typechecker.Ty::Var(a),
      @typechecker.Ty::arrow(@typechecker.Ty::Var(a), @typechecker.Ty::Var(a)),
    ),
  )
  self.insert_builtin(
    ["$max", "$min", "$step", "$mod", "$pow", "$reflect", "$atan"],
    binary_numeric,
  )
  // $clamp : a -> a -> a -> a
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["$clamp"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(a),
          @typechecker.Ty::arrow(
            @typechecker.Ty::Var(a),
            @typechecker.Ty::Var(a),
          ),
        ),
      ),
    ),
  )
  // $mix / $smoothstep : a -> a -> b -> a
  let a = fresh_var_id(self.engine)
  let b = fresh_var_id(self.engine)
  self.insert_builtin(
    ["$mix", "$smoothstep"],
    @typechecker.Scheme::poly(
      [a, b],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(a),
          @typechecker.Ty::arrow(
            @typechecker.Ty::Var(b),
            @typechecker.Ty::Var(a),
          ),
        ),
      ),
    ),
  )
  // $length : Vec n a -> a
  let dim = fresh_var_id(self.engine)
  let scalar = fresh_var_id(self.engine)
  let vec_ty = @typechecker.Ty::app(
    @typechecker.Ty::app(@typechecker.Ty::Con("Vec"), @typechecker.Ty::Var(dim)),
    @typechecker.Ty::Var(scalar),
  )
  self.insert_builtin(
    ["$length"],
    @typechecker.Scheme::poly(
      [dim, scalar],
      @typechecker.Ty::arrow(vec_ty, @typechecker.Ty::Var(scalar)),
    ),
  )
  // $dot : Vec n a -> Vec n a -> a
  let dim = fresh_var_id(self.engine)
  let scalar = fresh_var_id(self.engine)
  let vec_ty = @typechecker.Ty::app(
    @typechecker.Ty::app(@typechecker.Ty::Con("Vec"), @typechecker.Ty::Var(dim)),
    @typechecker.Ty::Var(scalar),
  )
  self.insert_builtin(
    ["$dot"],
    @typechecker.Scheme::poly(
      [dim, scalar],
      @typechecker.Ty::arrow(
        vec_ty,
        @typechecker.Ty::arrow(vec_ty, @typechecker.Ty::Var(scalar)),
      ),
    ),
  )
  // $vec2 : a -> a -> Vec 2 a
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["$vec2"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(a),
          @typechecker.vector_ty(2UL, @typechecker.Ty::Var(a)),
        ),
      ),
    ),
  )
  // $vec3 : a -> a -> a -> Vec 3 a
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["$vec3"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(a),
          @typechecker.Ty::arrow(
            @typechecker.Ty::Var(a),
            @typechecker.vector_ty(3UL, @typechecker.Ty::Var(a)),
          ),
        ),
      ),
    ),
  )
  // $vec4 : a -> a -> a -> a -> Vec 4 a
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["$vec4"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.Ty::arrow(
          @typechecker.Ty::Var(a),
          @typechecker.Ty::arrow(
            @typechecker.Ty::Var(a),
            @typechecker.Ty::arrow(
              @typechecker.Ty::Var(a),
              @typechecker.vector_ty(4UL, @typechecker.Ty::Var(a)),
            ),
          ),
        ),
      ),
    ),
  )
  // $splat2/3/4 : a -> Vec N a
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["$splat2"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.vector_ty(2UL, @typechecker.Ty::Var(a)),
      ),
    ),
  )
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["$splat3"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.vector_ty(3UL, @typechecker.Ty::Var(a)),
      ),
    ),
  )
  let a = fresh_var_id(self.engine)
  self.insert_builtin(
    ["$splat4"],
    @typechecker.Scheme::poly(
      [a],
      @typechecker.Ty::arrow(
        @typechecker.Ty::Var(a),
        @typechecker.vector_ty(4UL, @typechecker.Ty::Var(a)),
      ),
    ),
  )
  // $vecX/Y/Z/W : Vec n a -> a
  let dim = fresh_var_id(self.engine)
  let scalar = fresh_var_id(self.engine)
  let vec_ty = @typechecker.Ty::app(
    @typechecker.Ty::app(@typechecker.Ty::Con("Vec"), @typechecker.Ty::Var(dim)),
    @typechecker.Ty::Var(scalar),
  )
  let vec_proj = @typechecker.Scheme::poly(
    [dim, scalar],
    @typechecker.Ty::arrow(vec_ty, @typechecker.Ty::Var(scalar)),
  )
  self.insert_builtin(["$vecX", "$vecY", "$vecZ", "$vecW"], vec_proj)
}

// ---------------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------------

///|
pub fn SemanticAnalyzer::analyze(
  self : SemanticAnalyzer,
  prog : @ast.Program
) -> Unit {
  // Pass 1: register data type declarations.
  for d in prog.decls {
    match d {
      DataDecl(name, type_params, ctors, span) =>
        self.register_data_decl(name, type_params, ctors, span)
      _ => ()
    }
  }
  // Pass 1b: type aliases.
  for d in prog.decls {
    match d {
      TypeAlias(name, _params, ty, _) => {
        let scheme = self.convert_syntax_type(ty)
        self.env.insert(name, scheme)
      }
      _ => ()
    }
  }
  // Pass 2: type signatures.
  for d in prog.decls {
    match d {
      TypeSig(name, ty, _) => {
        let scheme = self.convert_syntax_type(ty)
        self.env.insert(name, scheme)
      }
      _ => ()
    }
  }
  // Pass 3: type-check function bodies.
  for d in prog.decls {
    match d {
      FunDecl(name, params, body, where_binds, span) =>
        self.check_function(name, params, body, where_binds, span)
      EntryPoint(_, name, params, body, span) =>
        self.check_function(name, params, body, Array::new(), span)
      _ => ()
    }
  }
}

///|
fn SemanticAnalyzer::register_data_decl(
  self : SemanticAnalyzer,
  name : String,
  type_params : Array[String],
  ctors : Array[@ast.ConDecl],
  _span : @span.Span
) -> Unit {
  let scope : Map[String, UInt] = Map::new()
  for p in type_params {
    scope[p] = fresh_var_id(self.engine)
  }
  let scheme_vars = type_params_to_vars(scope, type_params)
  let result_ty = apply_type_params(name, type_params, scope)
  let constructor_names : Array[String] = Array::new()
  let mut tag : UInt = 0
  for con in ctors {
    constructor_names.push(con.name)
    let (con_ty, field_info) = build_constructor_ty(self, con.fields, result_ty, scope)
    self.constructors[con.name] = @typechecker.ConstructorInfo::{
      type_name: name,
      tag,
      scheme_vars,
      fields: field_info,
      result_ty,
    }
    self.env.insert(con.name, @typechecker.Scheme::poly(scheme_vars, con_ty))
    tag = tag + 1
  }
  self.data_types[name] = DataTypeInfo::{
    name,
    type_params,
    constructors: constructor_names,
  }
}

///|
fn type_params_to_vars(
  scope : Map[String, UInt],
  type_params : Array[String]
) -> Array[UInt] {
  let out : Array[UInt] = Array::new()
  for p in type_params {
    match scope.get(p) {
      Some(v) => out.push(v)
      None => ()
    }
  }
  out
}

///|
fn apply_type_params(
  name : String,
  type_params : Array[String],
  scope : Map[String, UInt]
) -> @typechecker.Ty {
  let mut ty : @typechecker.Ty = @typechecker.Ty::Con(name)
  for p in type_params {
    match scope.get(p) {
      Some(v) => ty = @typechecker.Ty::App(ty, @typechecker.Ty::Var(v))
      None => ()
    }
  }
  ty
}

///|
fn build_constructor_ty(
  sa : SemanticAnalyzer,
  fields : @ast.ConFields,
  result_ty : @typechecker.Ty,
  scope : Map[String, UInt]
) -> (@typechecker.Ty, @typechecker.ConstructorFields) {
  match fields {
    Empty => (result_ty, @typechecker.ConstructorFields::Empty)
    Positional(tys) => {
      let field_tys : Array[@typechecker.Ty] = Array::new()
      for t in tys {
        field_tys.push(sa.convert_syntax_type_with_scope(t, scope))
      }
      let ty = arrow_fold(field_tys, result_ty)
      (ty, @typechecker.ConstructorFields::Positional(field_tys))
    }
    RecordFields(rs) => {
      let field_tys : Array[@typechecker.Ty] = Array::new()
      let record_fields : Array[(String, @typechecker.Ty)] = Array::new()
      for r in rs {
        let (n, t) = r
        let converted = sa.convert_syntax_type_with_scope(t, scope)
        field_tys.push(converted)
        record_fields.push((n, converted))
      }
      let ty = arrow_fold(field_tys, result_ty)
      (ty, @typechecker.ConstructorFields::RecordFields(record_fields))
    }
  }
}

///|
fn SemanticAnalyzer::convert_syntax_type(
  self : SemanticAnalyzer,
  ty : @ast.Type
) -> @typechecker.Scheme {
  let scope : Map[String, UInt] = Map::new()
  let t = self.convert_syntax_type_with_scope(ty, scope)
  let vars : Array[UInt] = Array::new()
  scope.each(fn(_, v) { vars.push(v) })
  @typechecker.Scheme::poly(vars, t)
}

///|
fn SemanticAnalyzer::convert_syntax_type_with_scope(
  self : SemanticAnalyzer,
  ty : @ast.Type,
  scope : Map[String, UInt]
) -> @typechecker.Ty {
  let raw = match ty {
    Con(name, _) => @typechecker.Ty::Con(name)
    Var(name, _) => {
      let id = match scope.get(name) {
        Some(v) => v
        None => {
          let v = fresh_var_id(self.engine)
          scope[name] = v
          v
        }
      }
      @typechecker.Ty::Var(id)
    }
    Nat(n, _) => @typechecker.Ty::Nat(n)
    Arrow(a, b, _) =>
      @typechecker.Ty::arrow(
        self.convert_syntax_type_with_scope(a, scope),
        self.convert_syntax_type_with_scope(b, scope),
      )
    App(f, a, _) =>
      @typechecker.Ty::App(
        self.convert_syntax_type_with_scope(f, scope),
        self.convert_syntax_type_with_scope(a, scope),
      )
    Paren(inner, _) => self.convert_syntax_type_with_scope(inner, scope)
    Tuple(elems, _) =>
      if elems.length() == 0 {
        @typechecker.Ty::unit()
      } else {
        self.convert_syntax_type_with_scope(elems[0], scope)
      }
    Unit(_) => @typechecker.Ty::unit()
  }
  @typechecker.normalize_type_aliases(raw)
}

///|
fn SemanticAnalyzer::check_function(
  self : SemanticAnalyzer,
  name : String,
  params : Array[@ast.Pat],
  body : @ast.Expr,
  where_binds : Array[(String, @ast.Expr)],
  span : @span.Span
) -> Unit {
  let local_env = clone_env(self.env)
  let param_types : Array[@typechecker.Ty] = Array::new()
  for pat in params {
    let ty = self.engine.fresh_var()
    self.bind_pattern(pat, ty, local_env)
    param_types.push(ty)
  }
  let body_ast = if where_binds.length() == 0 {
    body
  } else {
    @ast.Expr::Let(where_binds, body, span)
  }
  let body_ty = self.infer_expr(body_ast, local_env)
  let mut fun_ty = body_ty
  let mut i = param_types.length() - 1
  while i >= 0 {
    fun_ty = @typechecker.Ty::arrow(param_types[i], fun_ty)
    i = i - 1
  }
  match self.env.lookup(name) {
    Some(scheme) => {
      let declared = self.engine.instantiate(scheme)
      self.engine.unify(fun_ty, declared, span)
    }
    None => {
      let scheme = self.engine.generalize(self.env, fun_ty)
      self.env.insert(name, scheme)
    }
  }
}

///|
fn clone_env(src : @typechecker.TypeEnv) -> @typechecker.TypeEnv {
  let dst = @typechecker.TypeEnv::new()
  src.bindings.each(fn(k, v) { dst.insert(k, v) })
  dst
}

///|
fn SemanticAnalyzer::bind_pattern(
  self : SemanticAnalyzer,
  pat : @ast.Pat,
  ty : @typechecker.Ty,
  env : @typechecker.TypeEnv
) -> Unit {
  match pat {
    Var(name, _) => env.insert(name, @typechecker.Scheme::mono(ty))
    Wild(_) => ()
    Con(name, sub_pats, span) =>
      match self.constructors.get(name) {
        Some(info) => {
          let inst = info.instantiate(self.engine)
          self.engine.unify(ty, inst.result_ty, span)
          match inst.fields {
            Positional(field_tys) => {
              let n = if field_tys.length() < sub_pats.length() {
                field_tys.length()
              } else {
                sub_pats.length()
              }
              let mut i = 0
              while i < n {
                self.bind_pattern(sub_pats[i], field_tys[i], env)
                i = i + 1
              }
            }
            Empty => ()
            RecordFields(fields) => {
              let n = if fields.length() < sub_pats.length() {
                fields.length()
              } else {
                sub_pats.length()
              }
              let mut i = 0
              while i < n {
                let (_, fty) = fields[i]
                self.bind_pattern(sub_pats[i], fty, env)
                i = i + 1
              }
            }
          }
        }
        None =>
          self.engine.diagnostics().push(
            @diagnostics.Diagnostic::error("Unknown constructor: " + name)
              .with_label(@diagnostics.Label::primary(span, "not found"))
              .with_help(
                "declare it in a data definition before using it in a pattern",
              ),
          )
      }
    Lit(lit, span) => self.engine.unify(ty, lit_type(lit), span)
    Paren(inner, _) => self.bind_pattern(inner, ty, env)
    Tuple(pats, _) =>
      if pats.length() > 0 {
        self.bind_pattern(pats[0], ty, env)
      }
    Record(con_name, fields, span) =>
      match self.constructors.get(con_name) {
        Some(info) => {
          let inst = info.instantiate(self.engine)
          self.engine.unify(ty, inst.result_ty, span)
          match inst.fields {
            RecordFields(con_fields) =>
              for field in fields {
                let (fname, maybe_pat) = field
                let fty = find_field_ty(con_fields, fname)
                match maybe_pat {
                  Some(p) => self.bind_pattern(p, fty, env)
                  None => env.insert(fname, @typechecker.Scheme::mono(fty))
                }
              }
            _ => ()
          }
        }
        None => ()
      }
    As(name, inner, _) => {
      env.insert(name, @typechecker.Scheme::mono(ty))
      self.bind_pattern(inner, ty, env)
    }
  }
}

///|
fn find_field_ty(
  fields : Array[(String, @typechecker.Ty)],
  name : String
) -> @typechecker.Ty {
  for f in fields {
    let (n, ty) = f
    if n == name {
      return ty
    }
  }
  @typechecker.Ty::Error
}

///|
fn lit_type(lit : @ast.Lit) -> @typechecker.Ty {
  match lit {
    Int(_) => @typechecker.Ty::i32()
    Float(_) => @typechecker.Ty::f32()
    StringLit(_) => @typechecker.Ty::Con("String")
    Char(_) => @typechecker.Ty::Con("Char")
  }
}

///|
fn SemanticAnalyzer::infer_expr(
  self : SemanticAnalyzer,
  expr : @ast.Expr,
  env : @typechecker.TypeEnv
) -> @typechecker.Ty {
  match expr {
    Lit(lit, _) => lit_type(lit)
    Var(name, span) =>
      match env.lookup(name) {
        Some(scheme) => self.engine.instantiate(scheme)
        None => {
          self.engine.diagnostics().push(
            @diagnostics.Diagnostic::error("Unbound variable: " + name)
              .with_label(@diagnostics.Label::primary(span, "not in scope"))
              .with_help(
                "bind the name in a parameter, let, where, or import declaration",
              ),
          )
          @typechecker.Ty::Error
        }
      }
    Con(name, span) =>
      match env.lookup(name) {
        Some(scheme) => self.engine.instantiate(scheme)
        None => {
          self.engine.diagnostics().push(
            @diagnostics.Diagnostic::error("Unknown constructor: " + name)
              .with_label(@diagnostics.Label::primary(span, "not in scope"))
              .with_help("declare it in a data definition before constructing it here"),
          )
          @typechecker.Ty::Error
        }
      }
    App(func, arg, span) => {
      let func_ty = self.infer_expr(func, env)
      let arg_ty = self.infer_expr(arg, env)
      let ret_ty = self.engine.fresh_var()
      self.engine.unify(func_ty, @typechecker.Ty::arrow(arg_ty, ret_ty), span)
      ret_ty
    }
    Infix(lhs, op, rhs, span) => {
      let op_ty : @typechecker.Ty = match env.lookup(op) {
        Some(scheme) => self.engine.instantiate(scheme)
        None => {
          self.engine.diagnostics().push(
            @diagnostics.Diagnostic::error("Unknown operator: " + op)
              .with_label(@diagnostics.Label::primary(span, "not in scope"))
              .with_help("operators are regular functions; define one or import it into scope"),
          )
          return @typechecker.Ty::Error
        }
      }
      let lhs_ty = self.infer_expr(lhs, env)
      let rhs_ty = self.infer_expr(rhs, env)
      let ret_ty = self.engine.fresh_var()
      self.engine.unify(
        op_ty,
        @typechecker.Ty::arrow(
          lhs_ty,
          @typechecker.Ty::arrow(rhs_ty, ret_ty),
        ),
        span,
      )
      ret_ty
    }
    Lambda(pats, body, _) => {
      let local = clone_env(env)
      let param_types : Array[@typechecker.Ty] = Array::new()
      for pat in pats {
        let ty = self.engine.fresh_var()
        self.bind_pattern(pat, ty, local)
        param_types.push(ty)
      }
      let body_ty = self.infer_expr(body, local)
      let mut result = body_ty
      let mut i = param_types.length() - 1
      while i >= 0 {
        result = @typechecker.Ty::arrow(param_types[i], result)
        i = i - 1
      }
      result
    }
    Let(binds, body, _) => {
      let local = clone_env(env)
      for pair in binds {
        let (name, e) = pair
        let ty = self.infer_expr(e, local)
        let scheme = self.engine.generalize(local, ty)
        local.insert(name, scheme)
      }
      self.infer_expr(body, local)
    }
    Case(scrutinee, arms, span) => {
      let scrut_ty = self.infer_expr(scrutinee, env)
      let result_ty = self.engine.fresh_var()
      for arm in arms {
        let (pat, body) = arm
        let arm_env = clone_env(env)
        self.bind_pattern(pat, scrut_ty, arm_env)
        let body_ty = self.infer_expr(body, arm_env)
        self.engine.unify(result_ty, body_ty, span)
      }
      result_ty
    }
    If(cond, then_expr, else_expr, span) => {
      let cond_ty = self.infer_expr(cond, env)
      self.engine.unify(cond_ty, @typechecker.Ty::bool(), span)
      let then_ty = self.infer_expr(then_expr, env)
      let else_ty = self.infer_expr(else_expr, env)
      self.engine.unify(then_ty, else_ty, span)
      then_ty
    }
    Paren(inner, _) => self.infer_expr(inner, env)
    Tuple(elems, _) =>
      if elems.length() == 0 {
        @typechecker.Ty::unit()
      } else {
        let tys : Array[@typechecker.Ty] = Array::new()
        for e in elems {
          tys.push(self.infer_expr(e, env))
        }
        tys[0]
      }
    Record(fields, _) => {
      for f in fields {
        let (_, e) = f
        let _ = self.infer_expr(e, env)
      }
      self.engine.fresh_var()
    }
    FieldAccess(inner, field, _) => {
      let base_ty = self.infer_expr(inner, env)
      let base_ty = self.engine.finalize(base_ty)
      if is_swizzle(field) {
        match extract_vec_type(base_ty) {
          Some(triple) => {
            let (n, scalar) = triple
            if validate_swizzle(field, n) {
              if field.length() == 1 {
                return scalar
              } else {
                return @typechecker.Ty::app(
                  @typechecker.Ty::app(
                    @typechecker.Ty::Con("Vec"),
                    @typechecker.Ty::Nat(field.length().to_uint64()),
                  ),
                  scalar,
                )
              }
            }
          }
          None => ()
        }
      }
      self.engine.fresh_var()
    }
    VecLit(elems, span) => {
      if elems.length() == 0 {
        self.engine.diagnostics().push(
          @diagnostics.Diagnostic::error("Empty vec literal")
            .with_label(@diagnostics.Label::primary(span, "needs at least 2 elements"))
            .with_help(
              "add vector elements so the scalar type and arity can be inferred",
            ),
        )
        return @typechecker.Ty::Error
      }
      let scalar_ty = self.engine.fresh_var()
      let mut total : UInt64 = 0UL
      for elem in elems {
        let elem_ty = self.infer_expr(elem, env)
        let elem_ty = self.engine.finalize(elem_ty)
        match extract_vec_type(elem_ty) {
          Some(triple) => {
            let (n, inner_scalar) = triple
            total = total + n.to_uint64()
            self.engine.unify(scalar_ty, inner_scalar, span)
          }
          None => {
            total = total + 1UL
            self.engine.unify(scalar_ty, elem_ty, span)
          }
        }
      }
      if total < 2UL || total > 4UL {
        self.engine.diagnostics().push(
          @diagnostics.Diagnostic::error(
            "Vec literal has " + total.to_string() +
            " components, expected 2, 3, or 4",
          )
            .with_label(@diagnostics.Label::primary(span, "invalid component count"))
            .with_help("WGSL vectors must have exactly 2, 3, or 4 scalar components"),
        )
        return @typechecker.Ty::Error
      }
      @typechecker.Ty::app(
        @typechecker.Ty::app(@typechecker.Ty::Con("Vec"), @typechecker.Ty::Nat(total)),
        scalar_ty,
      )
    }
    OpSection(op, span) =>
      match env.lookup(op) {
        Some(scheme) => self.engine.instantiate(scheme)
        None => {
          self.engine.diagnostics().push(
            @diagnostics.Diagnostic::error("Unknown operator: " + op)
              .with_label(@diagnostics.Label::primary(span, "not in scope"))
              .with_help(
                "operators are regular functions; define one or import it into scope",
              ),
          )
          @typechecker.Ty::Error
        }
      }
    Neg(inner, _) => self.infer_expr(inner, env)
    Do(stmts, _) => {
      let local = clone_env(env)
      let mut last_ty : @typechecker.Ty = @typechecker.Ty::unit()
      for stmt in stmts {
        match stmt {
          ExprStmt(e, _) => last_ty = self.infer_expr(e, local)
          Bind(name, e, _) => {
            let _ty = self.infer_expr(e, local)
            let inner_ty = self.engine.fresh_var()
            local.insert(name, @typechecker.Scheme::mono(inner_ty))
            last_ty = _ty
          }
          LetStmt(name, e, _) => {
            let ty = self.infer_expr(e, local)
            local.insert(name, @typechecker.Scheme::mono(ty))
          }
        }
      }
      last_ty
    }
  }
}

// ---------------------------------------------------------------------------
// Swizzle helpers
// ---------------------------------------------------------------------------

///|
/// Whether a field name is a valid swizzle pattern (xyzw or rgba, 1-4 chars).
pub fn is_swizzle(field : String) -> Bool {
  let len = field.length()
  if len == 0 || len > 4 {
    return false
  }
  let mut all_xyzw = true
  let mut all_rgba = true
  let mut i = 0
  while i < len {
    let c = field.unsafe_charcode_at(i)
    let xyzw = c == 0x78 || c == 0x79 || c == 0x7A || c == 0x77
    let rgba = c == 0x72 || c == 0x67 || c == 0x62 || c == 0x61
    if !xyzw {
      all_xyzw = false
    }
    if !rgba {
      all_rgba = false
    }
    i = i + 1
  }
  all_xyzw || all_rgba
}

///|
fn swizzle_index(c : Int) -> Int {
  match c {
    0x78 | 0x72 => 0 // 'x' | 'r'
    0x79 | 0x67 => 1 // 'y' | 'g'
    0x7A | 0x62 => 2 // 'z' | 'b'
    0x77 | 0x61 => 3 // 'w' | 'a'
    _ => 0
  }
}

///|
/// Validate that all swizzle components are within bounds for a Vec of size n.
pub fn validate_swizzle(field : String, n : UInt) -> Bool {
  let len = field.length()
  let mut i = 0
  while i < len {
    let c = field.unsafe_charcode_at(i)
    if swizzle_index(c).reinterpret_as_uint() >= n {
      return false
    }
    i = i + 1
  }
  true
}

///|
/// Extract Vec type info: `Vec n T` -> Some((n, T)).
pub fn extract_vec_type(ty : @typechecker.Ty) -> (UInt, @typechecker.Ty)? {
  let ty = @typechecker.normalize_type_aliases(ty)
  match ty {
    App(f, scalar) =>
      match f {
        App(con, nat) =>
          match (con, nat) {
            (Con(name), Nat(n)) =>
              if name == "Vec" {
                Some((n.to_uint(), scalar))
              } else {
                None
              }
            _ => None
          }
        _ => None
      }
    _ => None
  }
}
