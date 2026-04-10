/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

// Tree-sitter grammar for fwgsl — a pure functional language for WebGPU.
//
// fwgsl uses Haskell-style indentation-sensitive layout. Since tree-sitter
// has limited indentation support, this grammar uses a simplified approach:
// it treats the language as mostly flat at the top level and uses regex-based
// heuristics for nested constructs. For precise indentation, editors should
// rely on the LSP semantic tokens.

module.exports = grammar({
  name: "fwgsl",

  extras: ($) => [/\s/, $.line_comment, $.block_comment],

  word: ($) => $.identifier,

  conflicts: ($) => [
    [$.import_declaration],
    [$.constructor_expression, $.record_expression],
  ],

  rules: {
    source_file: ($) => repeat($._declaration),

    _declaration: ($) =>
      choice(
        $.module_declaration,
        $.import_declaration,
        $.data_declaration,
        $.type_alias,
        $.extern_declaration,
        $.trait_declaration,
        $.impl_declaration,
        $.bitfield_declaration,
        $.const_declaration,
        $.cfg_declaration,
        $.type_signature,
        $.function_declaration,
      ),

    // -- Module & imports ---------------------------------------------------

    module_declaration: ($) => seq("module", $.module_path),

    import_declaration: ($) =>
      seq(
        "import",
        $.module_path,
        optional(seq("as", $.upper_identifier)),
        optional(seq("when", $.cfg_predicate)),
      ),

    module_path: ($) =>
      seq($.upper_identifier, repeat(seq(".", $.upper_identifier))),

    // -- Data declarations --------------------------------------------------

    data_declaration: ($) =>
      seq(
        "data",
        field("name", $.upper_identifier),
        repeat($.type_variable),
        "=",
        $.constructor_list,
        optional(seq("deriving", $.deriving_list)),
      ),

    constructor_list: ($) =>
      seq($.constructor, repeat(seq("|", $.constructor))),

    constructor: ($) =>
      prec.right(seq(
        field("name", $.upper_identifier),
        optional(choice($.record_fields, repeat1($._simple_type))),
      )),

    record_fields: ($) =>
      seq("{", commaSep1($.record_field), optional(","), "}"),

    record_field: ($) =>
      seq(
        repeat($.attribute),
        field("name", $.identifier),
        ":",
        field("type", $._type),
      ),

    deriving_list: ($) =>
      seq("(", commaSep1($.upper_identifier), ")"),

    // -- Type alias ----------------------------------------------------------

    type_alias: ($) =>
      seq("alias", field("name", $.upper_identifier), "=", $._type),

    // -- Extern / resource ---------------------------------------------------

    extern_declaration: ($) =>
      prec.right(seq(
        "extern",
        "resource",
        field("name", $.identifier),
        ":",
        field("type", $._type),
        repeat($.attribute),
      )),

    // -- Trait / impl --------------------------------------------------------

    trait_declaration: ($) =>
      prec.right(seq(
        "trait",
        field("name", $.upper_identifier),
        repeat($.type_variable),
        "where",
        repeat($._trait_member),
      )),

    _trait_member: ($) =>
      choice($.type_signature, $.function_declaration),

    impl_declaration: ($) =>
      prec.right(seq(
        "impl",
        field("trait", $.upper_identifier),
        repeat1($._simple_type),
        "where",
        repeat($._impl_member),
      )),

    _impl_member: ($) =>
      choice($.type_signature, $.function_declaration),

    // -- Bitfield ------------------------------------------------------------

    bitfield_declaration: ($) =>
      seq(
        "bitfield",
        field("name", $.upper_identifier),
        ":",
        field("backing", $._type),
        "=",
        field("constructor", $.upper_identifier),
        "{",
        commaSep1($.bitfield_field),
        optional(","),
        "}",
      ),

    bitfield_field: ($) =>
      seq(
        field("name", $.identifier),
        ":",
        choice(
          // Bare integer width: `name : 5`
          field("bits", $.integer_literal),
          // Typed with explicit width: `name : Type : 5`
          seq(
            field("type", $.upper_identifier),
            ":",
            field("bits", $.integer_literal),
          ),
          // Bool or enum-inferred: `name : Bool` or `name : CapStyle`
          field("type", $.upper_identifier),
        ),
      ),

    // -- Const ---------------------------------------------------------------

    const_declaration: ($) =>
      seq(
        "const",
        field("name", $.identifier),
        optional(seq(":", $._type)),
        "=",
        $.expression,
      ),

    // -- Conditional compilation (when/cfg) ----------------------------------

    cfg_declaration: ($) =>
      prec.right(seq("when", $.cfg_predicate, repeat($._declaration))),

    cfg_predicate: ($) =>
      choice(
        $.cfg_feature,
        $.cfg_not,
        $.cfg_and,
        $.cfg_or,
        seq("(", $.cfg_predicate, ")"),
      ),

    cfg_feature: ($) => seq("cfg", ".", $.identifier),
    cfg_not: ($) => prec(3, seq("not", $.cfg_predicate)),
    cfg_and: ($) => prec.left(2, seq($.cfg_predicate, "&&", $.cfg_predicate)),
    cfg_or: ($) => prec.left(1, seq($.cfg_predicate, "||", $.cfg_predicate)),

    // -- Type signatures and function declarations ---------------------------

    type_signature: ($) =>
      seq(
        field("name", $.identifier),
        ":",
        field("type", $._type),
      ),

    function_declaration: ($) =>
      seq(
        repeat($.attribute),
        field("name", $.identifier),
        repeat($.pattern),
        "=",
        field("body", $.expression),
        optional($.where_clause),
      ),

    where_clause: ($) =>
      prec.right(seq("where", repeat1($.local_binding))),

    local_binding: ($) =>
      seq(
        field("name", $.identifier),
        repeat($.pattern),
        "=",
        $.expression,
      ),

    // -- Attributes ----------------------------------------------------------

    attribute: ($) =>
      seq(
        "@",
        $.identifier,
        optional(seq("(", commaSep1($.expression), ")")),
      ),

    // -- Types ---------------------------------------------------------------

    _type: ($) =>
      choice(
        $.function_type,
        $.forall_type,
        $._simple_type,
      ),

    function_type: ($) =>
      prec.right(1, seq($._simple_type, "->", $._type)),

    forall_type: ($) =>
      seq("forall", repeat1($.type_variable), ".", $._type),

    _simple_type: ($) =>
      choice(
        $.type_constructor,
        $.type_variable,
        $.type_application,
        $.tuple_type,
        $.unit_type,
        $.parenthesized_type,
      ),

    type_constructor: ($) => $.upper_identifier,

    type_variable: ($) => $.identifier,

    type_application: ($) =>
      seq($.upper_identifier, "<", commaSep1($._type), ">"),

    tuple_type: ($) =>
      seq("(", $._type, ",", commaSep1($._type), ")"),

    unit_type: ($) => seq("(", ")"),

    parenthesized_type: ($) => seq("(", $._type, ")"),

    // -- Patterns ------------------------------------------------------------

    pattern: ($) =>
      choice(
        $.identifier_pattern,
        $.constructor_pattern,
        $.wildcard_pattern,
        $.literal_pattern,
        $.tuple_pattern,
        $.parenthesized_pattern,
      ),

    identifier_pattern: ($) => $.identifier,
    wildcard_pattern: ($) => "_",
    literal_pattern: ($) => $._literal,

    constructor_pattern: ($) =>
      prec.right(seq($.upper_identifier, repeat($.pattern))),

    tuple_pattern: ($) =>
      seq("(", $.pattern, ",", commaSep1($.pattern), ")"),

    parenthesized_pattern: ($) => seq("(", $.pattern, ")"),

    // -- Expressions ---------------------------------------------------------

    expression: ($) =>
      choice(
        $.let_expression,
        $.if_expression,
        $.case_expression,
        $.match_expression,
        $.lambda_expression,
        $.loop_expression,
        $.do_expression,
        $.binary_expression,
        $.pipe_expression,
        $.dollar_expression,
        $._simple_expression,
      ),

    let_expression: ($) =>
      prec.right(-1, seq("let", repeat1($.let_binding), "in", $.expression)),

    let_binding: ($) =>
      seq(
        field("name", $.identifier),
        repeat($.pattern),
        "=",
        $.expression,
      ),

    if_expression: ($) =>
      prec.right(-1, seq("if", $.expression, "then", $.expression, "else", $.expression)),

    case_expression: ($) =>
      prec.right(-1, seq("case", $.expression, "of", repeat1($.case_arm))),

    match_expression: ($) =>
      prec.right(-1, seq("match", $.expression, repeat1($.match_arm))),

    case_arm: ($) =>
      prec.right(seq($.pattern, "->", $.expression)),

    match_arm: ($) =>
      prec.right(seq("|", $.pattern, "->", $.expression)),

    lambda_expression: ($) =>
      prec.right(-1, seq("\\", repeat1($.pattern), "->", $.expression)),

    loop_expression: ($) =>
      prec.right(-1, seq("loop", $.expression)),

    do_expression: ($) =>
      prec.right(-1, seq("do", $.expression)),

    binary_expression: ($) =>
      choice(
        ...[
          ["+", 6],
          ["-", 6],
          ["*", 7],
          ["/", 7],
          ["%", 7],
          ["==", 4],
          ["/=", 4],
          ["<", 4],
          [">", 4],
          ["<=", 4],
          [">=", 4],
          ["&&", 3],
          ["||", 2],
          ["::", 5],
        ].map(([op, prec_val]) =>
          prec.left(
            /** @type {number} */ (prec_val),
            seq(
              field("left", $.expression),
              field("operator", /** @type {string} */ (op)),
              field("right", $.expression),
            ),
          ),
        ),
      ),

    pipe_expression: ($) =>
      prec.left(1, seq($.expression, "|>", $.expression)),

    dollar_expression: ($) =>
      prec.right(0, seq($.expression, "$", $.expression)),

    _simple_expression: ($) =>
      choice(
        $.function_application,
        $._atomic_expression,
      ),

    function_application: ($) =>
      prec.left(10, seq($._simple_expression, $._atomic_expression)),

    _atomic_expression: ($) =>
      choice(
        $.identifier_expression,
        $.constructor_expression,
        $._literal,
        $.field_access,
        $.index_expression,
        $.list_expression,
        $.record_expression,
        $.record_update,
        $.tuple_expression,
        $.unit_expression,
        $.parenthesized_expression,
        $.builtin_identifier,
      ),

    identifier_expression: ($) => $.identifier,
    constructor_expression: ($) => $.upper_identifier,

    field_access: ($) =>
      prec.left(11, seq($._simple_expression, ".", $.identifier)),

    index_expression: ($) =>
      prec.left(11, seq($._simple_expression, "[", $.expression, "]")),

    list_expression: ($) =>
      seq("[", commaSep($.expression), "]"),

    record_expression: ($) =>
      seq($.upper_identifier, "{", commaSep1($.field_init), "}"),

    record_update: ($) =>
      seq($._simple_expression, "{", commaSep1($.field_init), "}"),

    field_init: ($) =>
      seq(field("name", $.identifier), "=", field("value", $.expression)),

    tuple_expression: ($) =>
      seq("(", $.expression, ",", commaSep1($.expression), ")"),

    unit_expression: ($) => seq("(", ")"),

    parenthesized_expression: ($) => seq("(", $.expression, ")"),

    // -- Identifiers and literals -------------------------------------------

    identifier: ($) => /[a-z_][a-zA-Z0-9_']*/,
    upper_identifier: ($) => /[A-Z][A-Za-z0-9_]*/,
    builtin_identifier: ($) => /\$[a-zA-Z_][a-zA-Z0-9_]*/,

    _literal: ($) =>
      choice(
        $.integer_literal,
        $.float_literal,
        $.string_literal,
        $.char_literal,
        $.boolean,
      ),

    integer_literal: ($) =>
      token(
        choice(
          /0[xX][0-9a-fA-F_]+[ui]?/,
          /0[oO][0-7_]+[ui]?/,
          /0[bB][01_]+[ui]?/,
          /[0-9][0-9_]*[ui]?/,
        ),
      ),

    float_literal: ($) =>
      token(/[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9_]+)?/),

    string_literal: ($) =>
      seq('"', repeat(choice(/[^"\\]+/, $.escape_sequence)), '"'),

    char_literal: ($) =>
      seq("'", choice(/[^'\\]/, $.escape_sequence), "'"),

    escape_sequence: ($) =>
      token.immediate(
        /\\(['"\\nrt0abfv]|x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\})/,
      ),

    boolean: ($) => choice("true", "false"),

    // -- Comments -----------------------------------------------------------

    line_comment: ($) => token(seq("--", /.*/)),

    block_comment: ($) =>
      token(seq("{-", /[\s\S]*?/, "-}")),
  },
});

/**
 * Comma-separated list (0 or more).
 * @param {RuleOrLiteral} rule
 */
function commaSep(rule) {
  return optional(commaSep1(rule));
}

/**
 * Comma-separated list (1 or more).
 * @param {RuleOrLiteral} rule
 */
function commaSep1(rule) {
  return seq(rule, repeat(seq(",", rule)));
}
