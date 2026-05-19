///|
/// `SyntaxKind` tags every token and CST node in fwgsl.
///
/// In the Rust crate variants are laid out as `#[repr(u8)]` so they can be
/// packed into arena-allocated green nodes; MoonBit doesn't expose
/// discriminants but offers `to_int` / `from_int` for the same round trip.
pub(all) enum SyntaxKind {
  // Trivia
  Whitespace
  Newline
  LineComment
  BlockComment

  // Layout (virtual tokens inserted by the layout resolver)
  LayoutBraceOpen
  LayoutSemicolon
  LayoutBraceClose

  // Punctuation
  LParen
  RParen
  LBracket
  RBracket
  LBrace
  RBrace
  Comma
  Semicolon
  Colon
  ColonColon
  Dot
  DotDot
  Arrow
  FatArrow
  Backslash
  At
  Pipe
  Equals
  Underscore

  // Operators
  Plus
  Minus
  Star
  Slash
  Percent
  Less
  Greater
  LessEqual
  GreaterEqual
  EqualEqual
  NotEqual
  AndAnd
  OrOr
  Bang
  Dollar
  Backtick
  PipeForward
  Compose
  LeftArrow

  // Literals
  IntLiteral
  FloatLiteral
  StringLiteral
  CharLiteral

  // Identifiers
  Ident
  UpperIdent
  Operator

  // Keywords
  KwModule
  KwWhere
  KwImport
  KwData
  KwType
  KwClass
  KwInstance
  KwLet
  KwIn
  KwCase
  KwOf
  KwMatch
  KwIf
  KwThen
  KwElse
  KwDo
  KwForall
  KwInfixl
  KwInfixr
  KwInfix
  KwDeriving

  // CST node kinds
  SourceFile
  ModuleDecl
  ModuleHeader
  ExportList
  ExportItem
  ImportDecl
  ImportList
  ImportItem
  DataDecl
  ConDecl
  FieldDecl
  TypeDecl
  ClassDecl
  InstanceDecl
  MethodDecl
  TypeSig
  FunDecl
  FunEquation
  GuardedExpr
  ParamPat
  WhereBind
  Attribute
  AttrArg

  // Expression nodes
  ExprApp
  ExprInfix
  ExprPrefix
  ExprLambda
  ExprLet
  ExprLetBind
  ExprCase
  ExprCaseArm
  ExprIf
  ExprDo
  ExprDoStmt
  ExprParen
  ExprTuple
  ExprList
  ExprRecord
  ExprFieldAccess
  ExprSection
  ExprLit
  ExprIdent
  ExprOp
  ExprCompose
  ExprBacktick

  // Type nodes
  TypeArrow
  TypeApp
  TypeCon
  TypeVar
  TypeParen
  TypeTuple
  TypeForall
  TypeConstraint

  // Pattern nodes
  PatWild
  PatVar
  PatCon
  PatLit
  PatParen
  PatTuple
  PatRecord
  PatAs
  PatFieldPat

  // Misc
  QualifiedName

  // Sentinel / error
  Error
  Eof
  Tombstone
} derive(Eq, Hash)

///|
/// Returns true for tokens the parser should skip when building the CST.
pub fn SyntaxKind::is_trivia(self : SyntaxKind) -> Bool {
  match self {
    Whitespace | Newline | LineComment | BlockComment => true
    _ => false
  }
}

///|
/// Returns true for any keyword variant (`Kw*`).
pub fn SyntaxKind::is_keyword(self : SyntaxKind) -> Bool {
  match self {
    KwModule
    | KwWhere
    | KwImport
    | KwData
    | KwType
    | KwClass
    | KwInstance
    | KwLet
    | KwIn
    | KwCase
    | KwOf
    | KwMatch
    | KwIf
    | KwThen
    | KwElse
    | KwDo
    | KwForall
    | KwInfixl
    | KwInfixr
    | KwInfix
    | KwDeriving => true
    _ => false
  }
}

///|
/// Returns true for literal tokens.
pub fn SyntaxKind::is_literal(self : SyntaxKind) -> Bool {
  match self {
    IntLiteral | FloatLiteral | StringLiteral | CharLiteral => true
    _ => false
  }
}

///|
/// Returns true for operator tokens.
///
/// The Rust crate computes this with a range check between `FIRST_OPERATOR`
/// (`Plus`) and `LAST_OPERATOR` (`PipeForward`); `Compose` and `LeftArrow`
/// appear after that range so they are deliberately excluded here as well.
pub fn SyntaxKind::is_operator(self : SyntaxKind) -> Bool {
  match self {
    Plus
    | Minus
    | Star
    | Slash
    | Percent
    | Less
    | Greater
    | LessEqual
    | GreaterEqual
    | EqualEqual
    | NotEqual
    | AndAnd
    | OrOr
    | Bang
    | Dollar
    | Backtick
    | PipeForward => true
    _ => false
  }
}

///|
/// Map a source string to its keyword `SyntaxKind`, if any.
pub fn keyword_from_str(s : String) -> SyntaxKind? {
  match s {
    "module" => Some(KwModule)
    "where" => Some(KwWhere)
    "import" => Some(KwImport)
    "data" => Some(KwData)
    "type" => Some(KwType)
    "class" => Some(KwClass)
    "instance" => Some(KwInstance)
    "let" => Some(KwLet)
    "in" => Some(KwIn)
    "case" => Some(KwCase)
    "of" => Some(KwOf)
    "match" => Some(KwMatch)
    "if" => Some(KwIf)
    "then" => Some(KwThen)
    "else" => Some(KwElse)
    "do" => Some(KwDo)
    "forall" => Some(KwForall)
    "infixl" => Some(KwInfixl)
    "infixr" => Some(KwInfixr)
    "infix" => Some(KwInfix)
    "deriving" => Some(KwDeriving)
    _ => None
  }
}

///|
/// Human-readable rendering used in diagnostics.
pub fn SyntaxKind::display(self : SyntaxKind) -> String {
  match self {
    Whitespace => "whitespace"
    Newline => "newline"
    LineComment => "line comment"
    BlockComment => "block comment"
    LayoutBraceOpen => "layout '{'"
    LayoutSemicolon => "layout ';'"
    LayoutBraceClose => "layout '}'"
    LParen => "'('"
    RParen => "')'"
    LBracket => "'['"
    RBracket => "']'"
    LBrace => "'{'"
    RBrace => "'}'"
    Comma => "','"
    Semicolon => "';'"
    Colon => "':'"
    ColonColon => "'::'"
    Dot => "'.'"
    DotDot => "'..'"
    Arrow => "'->'"
    FatArrow => "'=>'"
    Backslash => "'\\'"
    At => "'@'"
    Pipe => "'|'"
    Equals => "'='"
    Underscore => "'_'"
    Plus => "'+'"
    Minus => "'-'"
    Star => "'*'"
    Slash => "'/'"
    Percent => "'%'"
    Less => "'<'"
    Greater => "'>'"
    LessEqual => "'<='"
    GreaterEqual => "'>='"
    EqualEqual => "'=='"
    NotEqual => "'/='"
    AndAnd => "'&&'"
    OrOr => "'||'"
    Bang => "'!'"
    Dollar => "'$'"
    Backtick => "'`'"
    Compose => "'.'"
    LeftArrow => "'<-'"
    PipeForward => "'|>'"
    IntLiteral => "integer literal"
    FloatLiteral => "float literal"
    StringLiteral => "string literal"
    CharLiteral => "char literal"
    Ident => "identifier"
    UpperIdent => "upper identifier"
    Operator => "operator"
    KwModule => "'module'"
    KwWhere => "'where'"
    KwImport => "'import'"
    KwData => "'data'"
    KwType => "'type'"
    KwClass => "'class'"
    KwInstance => "'instance'"
    KwLet => "'let'"
    KwIn => "'in'"
    KwCase => "'case'"
    KwOf => "'of'"
    KwMatch => "'match'"
    KwIf => "'if'"
    KwThen => "'then'"
    KwElse => "'else'"
    KwDo => "'do'"
    KwForall => "'forall'"
    KwInfixl => "'infixl'"
    KwInfixr => "'infixr'"
    KwInfix => "'infix'"
    KwDeriving => "'deriving'"
    SourceFile => "source file"
    ModuleDecl => "module declaration"
    ModuleHeader => "module header"
    ExportList => "export list"
    ExportItem => "export item"
    ImportDecl => "import declaration"
    ImportList => "import list"
    ImportItem => "import item"
    DataDecl => "data declaration"
    ConDecl => "constructor declaration"
    FieldDecl => "field declaration"
    TypeDecl => "type alias declaration"
    ClassDecl => "class declaration"
    InstanceDecl => "instance declaration"
    MethodDecl => "method declaration"
    TypeSig => "type signature"
    FunDecl => "function declaration"
    FunEquation => "function equation"
    GuardedExpr => "guarded expression"
    ParamPat => "parameter pattern"
    WhereBind => "where binding"
    Attribute => "attribute"
    AttrArg => "attribute argument"
    ExprApp => "function application"
    ExprInfix => "infix expression"
    ExprPrefix => "prefix expression"
    ExprLambda => "lambda expression"
    ExprLet => "let expression"
    ExprLetBind => "let binding"
    ExprCase => "case expression"
    ExprCaseArm => "case arm"
    ExprIf => "if expression"
    ExprDo => "do expression"
    ExprDoStmt => "do statement"
    ExprParen => "parenthesized expression"
    ExprTuple => "tuple expression"
    ExprList => "list expression"
    ExprRecord => "record expression"
    ExprFieldAccess => "field access"
    ExprSection => "section expression"
    ExprLit => "literal expression"
    ExprIdent => "identifier expression"
    ExprOp => "operator expression"
    ExprCompose => "compose expression"
    ExprBacktick => "backtick expression"
    TypeArrow => "function type"
    TypeApp => "type application"
    TypeCon => "type constructor"
    TypeVar => "type variable"
    TypeParen => "parenthesized type"
    TypeTuple => "tuple type"
    TypeForall => "forall type"
    TypeConstraint => "type constraint"
    PatWild => "wildcard pattern"
    PatVar => "variable pattern"
    PatCon => "constructor pattern"
    PatLit => "literal pattern"
    PatParen => "parenthesized pattern"
    PatTuple => "tuple pattern"
    PatRecord => "record pattern"
    PatAs => "as pattern"
    PatFieldPat => "field pattern"
    QualifiedName => "qualified name"
    Error => "error"
    Eof => "end of file"
    Tombstone => "tombstone"
  }
}

///|
pub impl Show for SyntaxKind with output(self, logger) {
  logger.write_string(self.display())
}
