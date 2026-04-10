; fwgsl highlight queries for tree-sitter

; Keywords — control flow
["if" "then" "else" "case" "of" "match" "loop"] @keyword.control

; Keywords — storage/declaration
["let" "in" "const" "data" "alias" "extern" "resource"] @keyword.storage

; Keywords — other
["module" "import" "where" "trait" "impl" "bitfield"
 "do" "forall" "deriving" "as" "when" "cfg" "not"] @keyword

; Booleans
(boolean) @constant.builtin

; Attributes
(attribute "@" @punctuation.special (identifier) @attribute)

; Types (UpperIdent)
(type_constructor) @type
(type_application (upper_identifier) @type)
(upper_identifier) @type

; Type variables in type positions
(type_variable) @type

; Function definitions
(function_declaration name: (identifier) @function)
(type_signature name: (identifier) @function)

; Builtin function calls ($prefixed)
(builtin_identifier) @function.builtin

; Constructor patterns in case/match arms
(constructor_pattern (upper_identifier) @constructor)

; Constructor expressions
(constructor_expression) @constructor

; Constructors in data declarations
(constructor name: (upper_identifier) @constructor)

; Record fields
(record_field name: (identifier) @property)
(field_init name: (identifier) @property)
(field_access "." (identifier) @property)

; Parameters (identifiers in pattern position)
(identifier_pattern) @variable.parameter

; Variables
(identifier_expression) @variable

; Numeric literals
(integer_literal) @number
(float_literal) @number.float

; Strings
(string_literal) @string
(char_literal) @string
(escape_sequence) @string.escape

; Comments
(line_comment) @comment
(block_comment) @comment

; Operators
["+" "-" "*" "/" "%" "==" "/=" "<" ">" "<=" ">=" "&&" "||"] @operator
["|>" "$" "->" "=" "|" ":" "::" "."] @operator

; Punctuation
["(" ")" "[" "]" "{" "}"] @punctuation.bracket
[","] @punctuation.delimiter
["@"] @punctuation.special
