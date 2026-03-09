'use strict';

// ============================================================
// Example programs
// ============================================================
const EXAMPLES = {
    hello: `-- Hello World: basic functions
add : I32 -> I32 -> I32
add x y = x + y

double : I32 -> I32
double x = x * 2

square : I32 -> I32
square x = x * x

triple : I32 -> I32
triple x =
  let d = double x
      s = add d x
  in  s`,

    adt: `-- Algebraic Data Types and Pattern Matching
data Color = Red | Green | Blue

toI32 : Color -> I32
toI32 c = match c
  | Red   -> 0
  | Green -> 1
  | Blue  -> 2

isRed : Color -> I32
isRed c = match c
  | Red  -> 1
  | _    -> 0`,

    compute: `-- Compute Shader: double each element
@compute @workgroup_size(64, 1, 1)
main idx =
  let x = idx * 2
  in  x`,

    ifexpr: `-- If expressions and comparison
clamp : I32 -> I32 -> I32 -> I32
clamp lo hi x =
  if x < lo
    then lo
    else if x > hi
      then hi
      else x

abs : I32 -> I32
abs x = if x < 0 then 0 - x else x

sign : I32 -> I32
sign x =
  if x > 0 then 1
  else if x < 0 then 0 - 1
  else 0`,
};

const AUTO_COMPILE_DELAY_MS = 250;
const FWGSL_KEYWORD_SET = new Set([
    'module', 'where', 'import', 'data', 'type', 'class', 'instance',
    'let', 'in', 'case', 'of', 'match', 'if', 'then', 'else', 'do',
    'forall', 'infixl', 'infixr', 'infix', 'deriving',
]);

const STATIC_EDITOR_ITEMS = [
    {
        label: 'match',
        detail: 'Pattern matching expression',
        documentation: 'Branch on a value with pattern matching.\n\n```fwgsl\nmatch value\n  | Pattern -> result\n```',
        insertText: 'match ${1:value}\n  | ${2:pattern} -> ${3:result}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value'],
    },
    {
        label: 'let',
        detail: 'Let binding',
        documentation: 'Introduce local bindings and evaluate a body expression.\n\n```fwgsl\nlet x = value\nin body\n```',
        insertText: 'let ${1:name} = ${2:value}\nin ${3:body}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value'],
    },
    {
        label: 'if',
        detail: 'Conditional expression',
        documentation: 'Choose between two expressions based on a boolean condition.\n\n```fwgsl\nif condition\n  then when_true\n  else when_false\n```',
        insertText: 'if ${1:condition}\n  then ${2:when_true}\n  else ${3:when_false}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value'],
    },
    {
        label: 'where',
        detail: 'Local definitions',
        documentation: 'Attach helper bindings to a declaration.\n\n```fwgsl\nf x = body\nwhere\n  helper = x + 1\n```',
        insertText: 'where\n  ${1:name} = ${2:value}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value'],
    },
    {
        label: 'data',
        detail: 'Algebraic data type declaration',
        documentation: 'Declare an algebraic data type and its constructors.\n\n```fwgsl\ndata Option a = Some a | None\n```',
        insertText: 'data ${1:Type} ${2:a} = ${3:Constructor}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value', 'type'],
    },
    {
        label: 'type',
        detail: 'Type alias declaration',
        documentation: 'Define a named alias for a type expression.\n\n```fwgsl\ntype Vec4f = Vec 4 F32\n```',
        insertText: 'type ${1:Alias} = ${2:Type}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value', 'type'],
    },
    {
        label: 'class',
        detail: 'Type class declaration',
        documentation: 'Declare a type class interface.',
        insertText: 'class ${1:Class} ${2:f} where\n  ${3:member} : ${4:Type}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value', 'type'],
    },
    {
        label: 'instance',
        detail: 'Type class instance',
        documentation: 'Implement a type class for a concrete type.',
        insertText: 'instance ${1:Class} ${2:Type} where\n  ${3:member} = ${4:impl}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value', 'type'],
    },
    {
        label: 'module',
        detail: 'Module declaration',
        documentation: 'Declare the module name at the top of the file.',
        insertText: 'module ${1:Main}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value', 'type'],
    },
    {
        label: 'import',
        detail: 'Import declaration',
        documentation: 'Bring another module into scope.',
        insertText: 'import ${1:Module}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value', 'type'],
    },
    {
        label: 'do',
        detail: 'Do notation block',
        documentation: 'Sequence monadic statements in a block.',
        insertText: 'do\n  ${1:statement}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value'],
    },
    {
        label: 'forall',
        detail: 'Universal quantification',
        documentation: 'Bind type variables explicitly in a type expression.\n\n```fwgsl\nforall a. a -> a\n```',
        insertText: 'forall ${1:a}. ${2:type}',
        snippet: true,
        kind: 'Keyword',
        contexts: ['value', 'type'],
    },
    {
        label: 'I32',
        detail: '32-bit signed integer',
        documentation: 'Signed 32-bit integer type that lowers to `i32` in WGSL.',
        insertText: 'I32',
        snippet: false,
        kind: 'TypeParameter',
        contexts: ['value', 'type'],
    },
    {
        label: 'U32',
        detail: '32-bit unsigned integer',
        documentation: 'Unsigned 32-bit integer type that lowers to `u32` in WGSL.',
        insertText: 'U32',
        snippet: false,
        kind: 'TypeParameter',
        contexts: ['value', 'type'],
    },
    {
        label: 'F32',
        detail: '32-bit floating point',
        documentation: '32-bit floating point scalar that lowers to `f32` in WGSL.',
        insertText: 'F32',
        snippet: false,
        kind: 'TypeParameter',
        contexts: ['value', 'type'],
    },
    {
        label: 'Bool',
        detail: 'Boolean type',
        documentation: 'Boolean scalar type that lowers to `bool` in WGSL.',
        insertText: 'Bool',
        snippet: false,
        kind: 'TypeParameter',
        contexts: ['value', 'type'],
    },
    {
        label: 'Vec',
        detail: 'Vector type',
        documentation: 'Vector type parameterized by a type-level dimension and scalar.\n\n```fwgsl\nVec 3 F32\n```',
        insertText: 'Vec ${1:3} ${2:F32}',
        snippet: true,
        kind: 'TypeParameter',
        contexts: ['value', 'type'],
    },
    {
        label: 'Mat',
        detail: 'Matrix type',
        documentation: 'Matrix type parameterized by rows, columns, and scalar.\n\n```fwgsl\nMat 4 4 F32\n```',
        insertText: 'Mat ${1:4} ${2:4} ${3:F32}',
        snippet: true,
        kind: 'TypeParameter',
        contexts: ['value', 'type'],
    },
    {
        label: 'Array',
        detail: 'Fixed-size array type',
        documentation: 'Array type parameterized by a type-level length and element type.\n\n```fwgsl\nArray 16 F32\n```',
        insertText: 'Array ${1:16} ${2:F32}',
        snippet: true,
        kind: 'TypeParameter',
        contexts: ['value', 'type'],
    },
    {
        label: 'map',
        detail: 'Apply a function to each element',
        documentation: 'Functor-style mapping across a structure.',
        insertText: 'map',
        snippet: false,
        kind: 'Function',
        contexts: ['value'],
    },
    {
        label: 'foldl',
        detail: 'Left fold over a structure',
        documentation: 'Accumulate a result from left to right.',
        insertText: 'foldl',
        snippet: false,
        kind: 'Function',
        contexts: ['value'],
    },
    {
        label: 'foldr',
        detail: 'Right fold over a structure',
        documentation: 'Accumulate a result from right to left.',
        insertText: 'foldr',
        snippet: false,
        kind: 'Function',
        contexts: ['value'],
    },
    {
        label: 'id',
        detail: 'Identity function',
        documentation: 'Return the argument unchanged.',
        insertText: 'id',
        snippet: false,
        kind: 'Function',
        contexts: ['value'],
    },
    {
        label: 'compose',
        detail: 'Function composition',
        documentation: 'Compose two functions, equivalent to `(.)`.',
        insertText: 'compose',
        snippet: false,
        kind: 'Function',
        contexts: ['value'],
    },
    {
        label: 'pure',
        detail: 'Lift a value into an applicative',
        documentation: 'Inject a pure value into an applicative context.',
        insertText: 'pure',
        snippet: false,
        kind: 'Function',
        contexts: ['value'],
    },
    {
        label: 'bind',
        detail: 'Monadic bind',
        documentation: 'Sequence computations by feeding a result into the next step.',
        insertText: 'bind',
        snippet: false,
        kind: 'Function',
        contexts: ['value'],
    },
    {
        label: 'fmap',
        detail: 'Functor map',
        documentation: 'Map over values inside a functor.',
        insertText: 'fmap',
        snippet: false,
        kind: 'Function',
        contexts: ['value'],
    },
    {
        label: 'vertex',
        detail: 'Vertex entry point attribute',
        documentation: 'Mark a function as a vertex shader entry point.\n\n```fwgsl\n@vertex\nmain = ...\n```',
        insertText: 'vertex',
        snippet: false,
        kind: 'Property',
        contexts: ['attribute'],
    },
    {
        label: 'fragment',
        detail: 'Fragment entry point attribute',
        documentation: 'Mark a function as a fragment shader entry point.',
        insertText: 'fragment',
        snippet: false,
        kind: 'Property',
        contexts: ['attribute'],
    },
    {
        label: 'compute',
        detail: 'Compute entry point attribute',
        documentation: 'Mark a function as a compute shader entry point.',
        insertText: 'compute',
        snippet: false,
        kind: 'Property',
        contexts: ['attribute'],
    },
    {
        label: 'workgroup_size',
        detail: 'Workgroup size attribute',
        documentation: 'Set compute workgroup dimensions.\n\n```fwgsl\n@workgroup_size(64, 1, 1)\n```',
        insertText: 'workgroup_size(${1:64}, ${2:1}, ${3:1})',
        snippet: true,
        kind: 'Property',
        contexts: ['attribute'],
    },
    {
        label: 'group',
        detail: 'Bind group attribute',
        documentation: 'Specify the resource bind group index.',
        insertText: 'group(${1:0})',
        snippet: true,
        kind: 'Property',
        contexts: ['attribute'],
    },
    {
        label: 'binding',
        detail: 'Binding attribute',
        documentation: 'Specify the binding index within a bind group.',
        insertText: 'binding(${1:0})',
        snippet: true,
        kind: 'Property',
        contexts: ['attribute'],
    },
    {
        label: 'location',
        detail: 'Location attribute',
        documentation: 'Specify an input or output location.',
        insertText: 'location(${1:0})',
        snippet: true,
        kind: 'Property',
        contexts: ['attribute'],
    },
    {
        label: 'builtin',
        detail: 'Builtin attribute',
        documentation: 'Bind a parameter or field to a WGSL builtin.',
        insertText: 'builtin(${1:global_invocation_id})',
        snippet: true,
        kind: 'Property',
        contexts: ['attribute'],
    },
];

// ============================================================
// fwgsl language definition for Monaco
// ============================================================
const FWGSL_LANGUAGE = {
    defaultToken: '',
    tokenPostfix: '.fwgsl',
    keywords: [
        'module', 'where', 'import', 'data', 'type', 'class', 'instance',
        'let', 'in', 'case', 'of', 'match', 'if', 'then', 'else', 'do',
        'forall', 'infixl', 'infixr', 'infix', 'deriving',
    ],
    typeKeywords: [
        'I32', 'U32', 'F32', 'Bool', 'Vec', 'Mat', 'Array',
        'Vec2', 'Vec3', 'Vec4', 'Option', 'String',
    ],
    operators: [
        '->', '=>', '::', '..', '<-', '=',
        '+', '-', '*', '/', '%',
        '==', '/=', '<', '>', '<=', '>=',
        '&&', '||', '!', '$', '.', '|>',
        '|',
    ],
    symbols: /[=><!~?:&|+\-*\/\^%\.]+/,
    escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

    tokenizer: {
        root: [
            // Attributes
            [/@\w+/, 'annotation'],

            // Type/Constructor identifiers (uppercase start)
            [/[A-Z][\w']*/, {
                cases: {
                    '@typeKeywords': 'type.identifier',
                    '@default': 'constructor'
                }
            }],

            // Builtin identifiers ($sin, $vec4, etc.)
            [/\$[a-zA-Z_][\w']*/, 'builtin'],

            // Keywords and identifiers
            [/[a-z_][\w']*/, {
                cases: {
                    '@keywords': 'keyword',
                    '@default': 'identifier'
                }
            }],

            // Whitespace
            { include: '@whitespace' },

            // Delimiters
            [/[{}()\[\]]/, '@brackets'],
            [/[,;]/, 'delimiter'],

            // Operators
            [/@symbols/, {
                cases: {
                    '@operators': 'operator',
                    '@default': ''
                }
            }],

            // Numbers
            [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
            [/0[xX][0-9a-fA-F]+/, 'number.hex'],
            [/0[oO][0-7]+/, 'number.octal'],
            [/0[bB][01]+/, 'number.binary'],
            [/\d+/, 'number'],

            // Strings
            [/"([^"\\]|\\.)*$/, 'string.invalid'],
            [/"/, { token: 'string.quote', bracket: '@open', next: '@string' }],

            // Characters
            [/'[^\\']'/, 'string.char'],
            [/(')(@escapes)(')/, ['string.char', 'string.escape', 'string.char']],

            // Backtick infix
            [/`[a-z][\w']*`/, 'operator.infix'],
        ],

        string: [
            [/[^\\"]+/, 'string'],
            [/@escapes/, 'string.escape'],
            [/\\./, 'string.escape.invalid'],
            [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }],
        ],

        whitespace: [
            [/[ \t\r\n]+/, 'white'],
            [/\{-/, 'comment', '@comment'],
            [/--.*$/, 'comment'],
        ],

        comment: [
            [/[^{-]+/, 'comment'],
            [/\{-/, 'comment', '@push'],
            [/-\}/, 'comment', '@pop'],
            [/[{-]/, 'comment'],
        ],
    },
};

// ============================================================
// Monaco theme
// ============================================================
const FWGSL_THEME = {
    base: 'vs-dark',
    inherit: true,
    rules: [
        { token: 'keyword', foreground: 'c678dd', fontStyle: 'bold' },
        { token: 'type.identifier', foreground: 'e5c07b' },
        { token: 'constructor', foreground: 'e5c07b' },
        { token: 'identifier', foreground: 'abb2bf' },
        { token: 'number', foreground: 'd19a66' },
        { token: 'number.float', foreground: 'd19a66' },
        { token: 'number.hex', foreground: 'd19a66' },
        { token: 'string', foreground: '98c379' },
        { token: 'string.char', foreground: '98c379' },
        { token: 'string.escape', foreground: '56b6c2' },
        { token: 'comment', foreground: '5c6370', fontStyle: 'italic' },
        { token: 'operator', foreground: '56b6c2' },
        { token: 'operator.infix', foreground: '56b6c2', fontStyle: 'italic' },
        { token: 'builtin', foreground: '61afef', fontStyle: 'italic' },
        { token: 'annotation', foreground: '61afef' },
        { token: 'delimiter', foreground: 'abb2bf' },
        { token: 'bracket', foreground: 'abb2bf' },
    ],
    colors: {
        'editor.background': '#111118',
        'editor.foreground': '#abb2bf',
        'editor.lineHighlightBackground': '#1a1a2420',
        'editor.selectionBackground': '#3e4451',
        'editorCursor.foreground': '#6366f1',
        'editorLineNumber.foreground': '#3a3a4a',
        'editorLineNumber.activeForeground': '#6366f1',
        'editorIndentGuide.background': '#1e1e2e',
        'editorIndentGuide.activeBackground': '#2a2a3a',
        'editor.selectionHighlightBackground': '#3e445140',
        'editorBracketMatch.background': '#6366f120',
        'editorBracketMatch.border': '#6366f150',
    },
};

// ============================================================
// WGSL language for output panel (simplified)
// ============================================================
const WGSL_LANGUAGE = {
    defaultToken: '',
    tokenPostfix: '.wgsl',
    keywords: [
        'fn', 'let', 'var', 'return', 'if', 'else', 'for', 'while',
        'loop', 'break', 'continue', 'struct', 'true', 'false',
    ],
    typeKeywords: [
        'i32', 'u32', 'f32', 'bool', 'vec2', 'vec3', 'vec4',
        'mat2x2', 'mat3x3', 'mat4x4', 'array',
    ],
    tokenizer: {
        root: [
            [/@\w+/, 'annotation'],
            [/[a-zA-Z_]\w*/, {
                cases: {
                    '@keywords': 'keyword',
                    '@typeKeywords': 'type',
                    '@default': 'identifier',
                }
            }],
            [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
            [/\d+[iu]?/, 'number'],
            [/"[^"]*"/, 'string'],
            [/\/\/.*$/, 'comment'],
            [/[{}()\[\]]/, '@brackets'],
            [/[<>]=?|[!=]=|[+\-*\/%&|^]/, 'operator'],
        ],
    },
};

const WGSL_THEME_RULES = [
    { token: 'keyword.wgsl', foreground: 'ff79c6' },
    { token: 'type.wgsl', foreground: '8be9fd' },
    { token: 'annotation.wgsl', foreground: '50fa7b' },
    { token: 'number.wgsl', foreground: 'bd93f9' },
    { token: 'number.float.wgsl', foreground: 'bd93f9' },
    { token: 'comment.wgsl', foreground: '6272a4', fontStyle: 'italic' },
    { token: 'string.wgsl', foreground: 'f1fa8c' },
];

function registerEditorProviders(monaco) {
    monaco.languages.registerCompletionItemProvider('fwgsl', {
        triggerCharacters: ['@', '.', ':', ' '],
        provideCompletionItems(model, position) {
            return {
                suggestions: buildEditorSuggestions(monaco, model, position),
            };
        },
    });

    monaco.languages.registerHoverProvider('fwgsl', {
        provideHover(model, position) {
            const word = model.getWordAtPosition(position);
            if (!word) {
                return null;
            }

            const isAttribute = isAttributeContext(model, position, word.word);
            const staticItem = lookupStaticEditorItem(word.word, isAttribute);
            const range = new monaco.Range(
                position.lineNumber,
                word.startColumn,
                position.lineNumber,
                word.endColumn,
            );

            if (staticItem) {
                return {
                    range,
                    contents: [
                        {
                            value: `**\`${word.word}\`**\n\n${staticItem.documentation}`,
                        },
                    ],
                };
            }

            const symbols = collectDocumentSymbols(model.getValue());
            if (symbols.signatures.has(word.word)) {
                return {
                    range,
                    contents: [
                        {
                            value: `\`\`\`fwgsl\n${word.word} : ${symbols.signatures.get(word.word)}\n\`\`\`\n\nLocal binding from this document.`,
                        },
                    ],
                };
            }
            if (symbols.constructors.has(word.word)) {
                return {
                    range,
                    contents: [
                        {
                            value: `**\`${word.word}\`**\n\nConstructor for \`${symbols.constructors.get(word.word)}\`.`,
                        },
                    ],
                };
            }
            if (symbols.types.has(word.word)) {
                return {
                    range,
                    contents: [
                        {
                            value: `**\`${word.word}\`**\n\nUser-defined data type from this document.`,
                        },
                    ],
                };
            }

            return null;
        },
    });
}

function buildEditorSuggestions(monaco, model, position) {
    const word = model.getWordUntilPosition(position);
    const prefix = word.word || '';
    const context = getEditorCompletionContext(model, position, prefix);
    const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
    };
    const seen = new Set();
    const suggestions = [];

    for (const item of STATIC_EDITOR_ITEMS) {
        if (!item.contexts.includes(context)) {
            continue;
        }
        if (prefix && !item.label.startsWith(prefix)) {
            continue;
        }

        suggestions.push({
            label: item.label,
            kind: toMonacoCompletionKind(monaco, item.kind),
            detail: item.detail,
            documentation: {
                value: item.documentation,
            },
            insertText: item.insertText,
            insertTextRules: item.snippet
                ? monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
                : undefined,
            range,
            sortText: `1-${item.label}`,
        });
        seen.add(item.label);
    }

    if (context === 'attribute') {
        return suggestions;
    }

    const symbols = collectDocumentSymbols(model.getValue());
    for (const identifier of symbols.identifiers) {
        if (seen.has(identifier)) {
            continue;
        }
        if (prefix && !identifier.startsWith(prefix)) {
            continue;
        }
        if (context === 'type' && !/^[A-Z]/.test(identifier)) {
            continue;
        }

        suggestions.push({
            label: identifier,
            kind: inferDocumentSymbolKind(monaco, symbols, identifier),
            detail: inferDocumentSymbolDetail(symbols, identifier),
            documentation: {
                value: inferDocumentSymbolDocumentation(symbols, identifier),
            },
            insertText: identifier,
            range,
            sortText: `0-${identifier}`,
        });
        seen.add(identifier);
    }

    return suggestions.sort((left, right) => {
        if (left.sortText === right.sortText) {
            return left.label.localeCompare(right.label);
        }
        return left.sortText.localeCompare(right.sortText);
    });
}

function toMonacoCompletionKind(monaco, kind) {
    return monaco.languages.CompletionItemKind[kind]
        ?? monaco.languages.CompletionItemKind.Text;
}

function getEditorCompletionContext(model, position, prefix) {
    if (isAttributeContext(model, position, prefix)) {
        return 'attribute';
    }
    if (/^[A-Z]/.test(prefix)) {
        return 'type';
    }

    const linePrefix = model.getLineContent(position.lineNumber).slice(0, position.column - 1);
    const beforePrefix = linePrefix.slice(0, Math.max(0, linePrefix.length - prefix.length));
    const lastColon = beforePrefix.lastIndexOf(':');
    const lastEquals = beforePrefix.lastIndexOf('=');

    if (lastColon !== -1 && lastColon > lastEquals) {
        return 'type';
    }
    if (/^\s*type\b/.test(linePrefix)) {
        return 'type';
    }

    return 'value';
}

function isAttributeContext(model, position, prefix) {
    const linePrefix = model.getLineContent(position.lineNumber).slice(0, position.column - 1);
    const beforePrefix = linePrefix.slice(0, Math.max(0, linePrefix.length - prefix.length));
    return beforePrefix.trimEnd().endsWith('@');
}

function lookupStaticEditorItem(label, isAttribute = false) {
    return STATIC_EDITOR_ITEMS.find((item) => {
        if (item.label !== label) {
            return false;
        }
        if (isAttribute) {
            return item.contexts.includes('attribute');
        }
        return !item.contexts.includes('attribute');
    }) ?? null;
}

function collectDocumentSymbols(source) {
    const signatures = new Map();
    const constructors = new Map();
    const types = new Set();
    const identifiers = new Set();

    for (const line of source.split('\n')) {
        const signatureMatch = line.match(/^\s*([A-Za-z_][\w']*)\s*:\s*(.+)$/);
        if (signatureMatch) {
            signatures.set(signatureMatch[1], signatureMatch[2].trim());
        }

        const dataMatch = line.match(/^\s*data\s+([A-Z][\w']*)(?:\s+[\w'\s]+)?\s*=\s*(.+)$/);
        if (dataMatch) {
            types.add(dataMatch[1]);
            for (const constructorPart of dataMatch[2].split('|')) {
                const constructorMatch = constructorPart.trim().match(/^([A-Z][\w']*)/);
                if (constructorMatch) {
                    constructors.set(constructorMatch[1], dataMatch[1]);
                }
            }
        }
    }

    for (const match of source.matchAll(/\b[A-Za-z_][\w']*\b/g)) {
        const identifier = match[0];
        if (!FWGSL_KEYWORD_SET.has(identifier)) {
            identifiers.add(identifier);
        }
    }

    return {
        signatures,
        constructors,
        types,
        identifiers,
    };
}

function inferDocumentSymbolKind(monaco, symbols, identifier) {
    if (symbols.constructors.has(identifier)) {
        return monaco.languages.CompletionItemKind.Constructor;
    }
    if (symbols.types.has(identifier) || /^[A-Z]/.test(identifier)) {
        return monaco.languages.CompletionItemKind.TypeParameter;
    }
    return monaco.languages.CompletionItemKind.Variable;
}

function inferDocumentSymbolDetail(symbols, identifier) {
    if (symbols.signatures.has(identifier)) {
        return `binding : ${symbols.signatures.get(identifier)}`;
    }
    if (symbols.constructors.has(identifier)) {
        return `constructor : ${symbols.constructors.get(identifier)}`;
    }
    if (symbols.types.has(identifier)) {
        return 'data type';
    }
    return 'identifier from this document';
}

function inferDocumentSymbolDocumentation(symbols, identifier) {
    if (symbols.signatures.has(identifier)) {
        return `\`\`\`fwgsl\n${identifier} : ${symbols.signatures.get(identifier)}\n\`\`\`\n\nLocal binding from this document.`;
    }
    if (symbols.constructors.has(identifier)) {
        return `**\`${identifier}\`**\n\nConstructor for \`${symbols.constructors.get(identifier)}\`.`;
    }
    if (symbols.types.has(identifier)) {
        return `**\`${identifier}\`**\n\nUser-defined data type from this document.`;
    }
    return '**Local identifier**';
}

// ============================================================
// App state
// ============================================================
let editor = null;
let wgslEditor = null;
let wasmModule = null;
let gpuDevice = null;
let pendingCompileHandle = null;
let diagnosticDecorationIds = [];

// ============================================================
// Initialize
// ============================================================
async function init() {
    await initMonaco();
    await initWasm();
    await initWebGPU();
    setupEventListeners();
    setupResizeHandlers();
    compile({ reason: 'initial' });
}

async function initMonaco() {
    return new Promise((resolve) => {
        require.config({
            paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs' }
        });
        require(['vs/editor/editor.main'], function (monaco) {
            window.monaco = monaco;

            // Register fwgsl language
            monaco.languages.register({ id: 'fwgsl' });
            monaco.languages.setMonarchTokensProvider('fwgsl', FWGSL_LANGUAGE);

            // Register WGSL language
            monaco.languages.register({ id: 'wgsl' });
            monaco.languages.setMonarchTokensProvider('wgsl', WGSL_LANGUAGE);

            // Define theme with both language rules
            monaco.editor.defineTheme('fwgsl-dark', {
                ...FWGSL_THEME,
                rules: [...FWGSL_THEME.rules, ...WGSL_THEME_RULES],
            });
            registerEditorProviders(monaco);

            // Create main editor
            editor = monaco.editor.create(document.getElementById('editor-container'), {
                value: EXAMPLES.hello,
                language: 'fwgsl',
                theme: 'fwgsl-dark',
                fontFamily: "'JetBrains Mono', 'Cascadia Code', monospace",
                fontSize: 13,
                lineHeight: 22,
                padding: { top: 12 },
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                smoothScrolling: true,
                cursorBlinking: 'smooth',
                cursorSmoothCaretAnimation: 'on',
                bracketPairColorization: { enabled: true },
                guides: {
                    bracketPairs: true,
                    indentation: true,
                },
                renderLineHighlight: 'all',
                renderWhitespace: 'selection',
                tabSize: 2,
                automaticLayout: true,
                quickSuggestions: {
                    other: true,
                    comments: false,
                    strings: false,
                },
                suggestOnTriggerCharacters: true,
                glyphMargin: true,
                hover: {
                    enabled: true,
                    delay: 150,
                    sticky: true,
                },
                overviewRulerBorder: false,
                hideCursorInOverviewRuler: true,
                scrollbar: {
                    verticalScrollbarSize: 8,
                    horizontalScrollbarSize: 8,
                },
            });

            // Create WGSL output editor (read-only)
            wgslEditor = monaco.editor.create(document.getElementById('wgsl-output'), {
                value: '-- Compile to see WGSL output',
                language: 'wgsl',
                theme: 'fwgsl-dark',
                fontFamily: "'JetBrains Mono', 'Cascadia Code', monospace",
                fontSize: 12,
                lineHeight: 20,
                padding: { top: 12 },
                readOnly: true,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                renderLineHighlight: 'none',
                lineNumbers: 'on',
                automaticLayout: true,
                overviewRulerBorder: false,
                scrollbar: {
                    verticalScrollbarSize: 8,
                    horizontalScrollbarSize: 8,
                },
            });

            // Track cursor position
            editor.onDidChangeCursorPosition((e) => {
                const pos = e.position;
                document.getElementById('status-line').textContent =
                    `Ln ${pos.lineNumber}, Col ${pos.column}`;
            });

            editor.onDidChangeModelContent(() => {
                scheduleCompile('typing');
            });

            resolve();
        });
    });
}

async function initWasm() {
    try {
        const pkg = await import('./pkg/fwgsl_wasm.js');
        await pkg.default();
        wasmModule = pkg;
        document.getElementById('editor-status').textContent = 'wasm ready';
    } catch (e) {
        console.warn('WASM module not available, using mock compiler', e);
        document.getElementById('editor-status').textContent = 'mock mode';
        // Mock compiler for development
        wasmModule = {
            compile: (source) => JSON.stringify({
                wgsl: generateMockWgsl(source),
                diagnostics: [],
            }),
            format: (source) => source,
            get_diagnostics: (source) => '[]',
        };
    }
}

/**
 * Generate mock WGSL output from fwgsl source for development/demo purposes.
 */
function generateMockWgsl(source) {
    const lines = source.split('\n');
    const functions = [];
    const types = [];

    for (const line of lines) {
        const fnMatch = line.match(/^(\w+)\s*:/);
        if (fnMatch && !line.startsWith('--')) {
            functions.push(fnMatch[1]);
        }
        const dataMatch = line.match(/^data\s+(\w+)/);
        if (dataMatch) {
            types.push(dataMatch[1]);
        }
    }

    let output = '// Generated WGSL from fwgsl compiler\n';
    output += '// fwgsl compiler WASM module not loaded.\n';
    output += '// Build with: mise run wasm\n';
    output += `// Source: ${lines.length} lines\n\n`;

    for (const t of types) {
        output += `struct ${t} {\n  tag: u32,\n  data: array<u32, 4>,\n}\n\n`;
    }

    for (const fn of functions) {
        if (fn === 'main') {
            output += `@compute @workgroup_size(64, 1, 1)\n`;
            output += `fn ${fn}(@builtin(global_invocation_id) gid: vec3<u32>) {\n`;
            output += `  // compiled from fwgsl\n`;
            output += `}\n\n`;
        } else {
            output += `fn ${fn}() -> i32 {\n`;
            output += `  // compiled from fwgsl\n`;
            output += `  return 0i;\n`;
            output += `}\n\n`;
        }
    }

    return output;
}

async function initWebGPU() {
    try {
        if (!navigator.gpu) {
            console.warn('WebGPU not available');
            return;
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return;
        gpuDevice = await adapter.requestDevice();
        console.log('WebGPU initialized');
    } catch (e) {
        console.warn('WebGPU initialization failed', e);
    }
}

/**
 * Run a WGSL compute shader on the GPU and visualize the output buffer on the canvas.
 * Creates a storage buffer, dispatches the compute shader, reads back results,
 * and renders them as colored pixels on the preview canvas.
 */
async function runComputePreview(wgslCode) {
    if (!gpuDevice) {
        showPreviewMessage('WebGPU not available');
        return;
    }

    // Only run if the shader has @compute
    if (!wgslCode.includes('@compute')) {
        showPreviewMessage('No @compute entry point found.\nCompile a compute shader to preview.');
        return;
    }

    try {
        const canvas = document.getElementById('webgpu-canvas');
        const overlay = document.getElementById('preview-overlay');
        const ctx = canvas.getContext('2d');
        const W = 256;
        const H = 256;
        canvas.width = W;
        canvas.height = H;

        const elementCount = W * H;
        const bufferSize = elementCount * 4; // i32 per element

        // Create storage buffer visible to the shader
        const storageBuffer = gpuDevice.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Buffer to read results back to CPU
        const readBuffer = gpuDevice.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Extract helper functions and main body from compiled WGSL,
        // then wrap with a storage buffer binding so we can read back results.
        const helperFns = [];
        const mainBody = [];
        let insideMain = false;
        let braceDepth = 0;

        for (const line of wgslCode.split('\n')) {
            if (line.match(/^@compute/)) {
                insideMain = true;
                continue;
            }
            if (insideMain && line.match(/^fn main/)) {
                braceDepth = 1;
                continue;
            }
            if (insideMain && braceDepth > 0) {
                braceDepth += (line.match(/\{/g) || []).length;
                braceDepth -= (line.match(/\}/g) || []).length;
                if (braceDepth <= 0) {
                    insideMain = false;
                } else {
                    mainBody.push(line);
                }
                continue;
            }
            if (!line.startsWith('//') && line.trim()) {
                helperFns.push(line);
            }
        }

        // Find the last let/var binding name in mainBody to use as the output value
        let lastVarName = null;
        for (let i = mainBody.length - 1; i >= 0; i--) {
            const m = mainBody[i].match(/^\s*(?:let|var)\s+(\w+)/);
            if (m) { lastVarName = m[1]; break; }
        }

        // Build a compute shader that writes the result to a storage buffer
        const wrappedShader = `
@group(0) @binding(0) var<storage, read_write> output: array<i32>;

${helperFns.join('\n')}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) _gid: vec3<u32>) {
  let _flat_idx = _gid.x + _gid.y * ${W}u;
  if (_flat_idx >= ${elementCount}u) { return; }
  let _gid_i32: i32 = i32(_flat_idx);
${mainBody.map(l => l.replace(/i32\(_gid\.x\)/, '_gid_i32')).join('\n')}
  output[_flat_idx] = ${lastVarName || '_gid_i32'};
}
`;

        let shaderModule;
        try {
            shaderModule = gpuDevice.createShaderModule({ code: wrappedShader });
            const info = await shaderModule.getCompilationInfo();
            const errors = info.messages.filter(m => m.type === 'error');
            if (errors.length > 0) {
                showPreviewMessage('GPU shader compile error:\n' + errors.map(e => e.message).join('\n'));
                return;
            }
        } catch (e) {
            showPreviewMessage('GPU shader error:\n' + e.message);
            return;
        }

        const bindGroupLayout = gpuDevice.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            }],
        });

        const pipelineLayout = gpuDevice.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        const pipeline = gpuDevice.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'main' },
        });

        const bindGroup = gpuDevice.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: storageBuffer },
            }],
        });

        // Dispatch
        const encoder = gpuDevice.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), 1);
        pass.end();
        encoder.copyBufferToBuffer(storageBuffer, 0, readBuffer, 0, bufferSize);
        gpuDevice.queue.submit([encoder.finish()]);

        // Read back
        await readBuffer.mapAsync(GPUMapMode.READ);
        const data = new Int32Array(readBuffer.getMappedRange());

        // Visualize as heatmap
        const imageData = ctx.createImageData(W, H);
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < data.length; i++) {
            minVal = Math.min(minVal, data[i]);
            maxVal = Math.max(maxVal, data[i]);
        }
        const range = maxVal - minVal || 1;

        for (let i = 0; i < data.length; i++) {
            const t = (data[i] - minVal) / range;
            const px = i * 4;
            // Viridis-inspired colormap
            imageData.data[px + 0] = Math.floor(lerp(68, 253, t));     // R
            imageData.data[px + 1] = Math.floor(lerp(1, 231, t * t));  // G
            imageData.data[px + 2] = Math.floor(lerp(84, 37, t));      // B
            imageData.data[px + 3] = 255;                               // A
        }

        ctx.putImageData(imageData, 0, 0);
        readBuffer.unmap();
        storageBuffer.destroy();
        readBuffer.destroy();

        overlay.style.display = 'none';

        // Show stats
        document.getElementById('preview-overlay').innerHTML = '';
    } catch (e) {
        showPreviewMessage('GPU error: ' + e.message);
        console.error('WebGPU preview error:', e);
    }
}

function lerp(a, b, t) {
    return a + (b - a) * Math.max(0, Math.min(1, t));
}

function showPreviewMessage(msg) {
    const overlay = document.getElementById('preview-overlay');
    overlay.style.display = 'flex';
    overlay.innerHTML = `
        <div class="preview-placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
            <p>${escapeHtml(msg)}</p>
        </div>`;
}

// ============================================================
// Compile
// ============================================================
function scheduleCompile(reason = 'typing') {
    if (!editor) {
        return;
    }

    document.getElementById('editor-status').textContent = 'dirty';
    window.clearTimeout(pendingCompileHandle);
    pendingCompileHandle = window.setTimeout(() => {
        compile({ reason });
    }, AUTO_COMPILE_DELAY_MS);
}

function compile(options = {}) {
    if (!wasmModule) {
        return;
    }

    window.clearTimeout(pendingCompileHandle);
    const source = editor.getValue();
    document.body.classList.add('compiling');
    document.getElementById('editor-status').textContent = 'compiling...';

    const startTime = performance.now();

    try {
        const resultJson = wasmModule.compile(source);
        const result = JSON.parse(resultJson);
        const elapsed = (performance.now() - startTime).toFixed(1);

        // Update WGSL output
        wgslEditor.setValue(result.wgsl || '// No output');

        // Update diagnostics
        updateDiagnostics(result.diagnostics || []);

        // Update status
        const issueCount = (result.diagnostics || []).length;
        const hasErrors = (result.diagnostics || []).some((diag) => diag.severity === 'error');
        document.getElementById('editor-status').textContent = options.reason === 'typing'
            ? hasErrors
                ? `live errors (${issueCount})`
                : 'live'
            : hasErrors
                ? `errors (${issueCount})`
                : 'compiled';
        document.getElementById('status-compile-time').textContent = `${elapsed}ms`;
        document.getElementById('output-status').textContent = issueCount > 0
            ? `${elapsed}ms · ${issueCount} issue${issueCount === 1 ? '' : 's'}`
            : `${elapsed}ms`;

        // Run WebGPU preview for compute shaders
        if (result.wgsl && !result.wgsl.startsWith('//')) {
            runComputePreview(result.wgsl);
        }

    } catch (e) {
        console.error('Compilation error:', e);
        updateDiagnostics([{
            severity: 'error',
            message: `Internal compiler error: ${e.message}`,
            line: 1,
            col: 1,
        }]);
        document.getElementById('editor-status').textContent = 'error';
    }

    document.body.classList.remove('compiling');
}

// ============================================================
// Format
// ============================================================
function formatSource() {
    try {
        const source = editor.getValue();
        const formatted = wasmModule.format(source);
        editor.setValue(formatted);
    } catch (e) {
        console.error('Format error:', e);
    }
}

// ============================================================
// Diagnostics
// ============================================================
function updateDiagnostics(diagnostics) {
    const list = document.getElementById('diagnostics-list');
    const badge = document.getElementById('diag-count');
    const summary = document.getElementById('diag-summary');

    if (!diagnostics || diagnostics.length === 0) {
        list.innerHTML = '<div class="diagnostics-empty"><span>No diagnostics</span></div>';
        badge.textContent = '0';
        badge.className = 'badge';
        summary.textContent = 'No issues';
        applyDiagnosticDecorations([]);
        return;
    }

    const counts = summarizeDiagnostics(diagnostics);
    badge.textContent = diagnostics.length;
    const hasErrors = counts.errors > 0;
    badge.className = `badge ${hasErrors ? 'has-errors' : 'has-warnings'}`;
    summary.textContent = formatDiagnosticSummary(counts);

    list.innerHTML = diagnostics.map(d => {
        const severity = d.severity || 'info';
        const icon = severity === 'error'
            ? '<svg class="diagnostic-icon error" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
            : severity === 'warning'
            ? '<svg class="diagnostic-icon warning" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            : '<svg class="diagnostic-icon info" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';

        return `
            <div class="diagnostic-item ${severity}" data-line="${d.line || 1}" data-col="${d.col || 1}">
                ${icon}
                <div class="diagnostic-body">
                    <span class="diagnostic-message">${escapeHtml(d.message)}</span>
                    ${renderDiagnosticMeta(d)}
                </div>
                <span class="diagnostic-location">${d.line || 1}:${d.col || 1}</span>
            </div>
        `;
    }).join('');

    // Click to jump to location
    list.querySelectorAll('.diagnostic-item').forEach(item => {
        item.addEventListener('click', () => {
            const line = parseInt(item.dataset.line);
            const col = parseInt(item.dataset.col);
            editor.revealLineInCenter(line);
            editor.setPosition({ lineNumber: line, column: col });
            editor.focus();
        });
    });

    // Set editor markers
    if (window.monaco) {
        const model = editor.getModel();
        const markers = diagnostics.map(d => ({
            severity: d.severity === 'error' ? monaco.MarkerSeverity.Error
                    : d.severity === 'warning' ? monaco.MarkerSeverity.Warning
                    : d.severity === 'hint' ? monaco.MarkerSeverity.Hint
                    : monaco.MarkerSeverity.Info,
            startLineNumber: d.line || 1,
            startColumn: d.col || 1,
            endLineNumber: d.endLine || d.line || 1,
            endColumn: d.endCol || (d.col || 1) + 1,
            message: formatDiagnosticMessage(d),
            source: 'fwgsl',
            code: d.code,
        }));
        monaco.editor.setModelMarkers(model, 'fwgsl', markers);
    }

    applyDiagnosticDecorations(diagnostics);
}

function summarizeDiagnostics(diagnostics) {
    return diagnostics.reduce((counts, diagnostic) => {
        const severity = diagnostic.severity || 'info';
        if (severity === 'error') {
            counts.errors += 1;
        } else if (severity === 'warning') {
            counts.warnings += 1;
        } else {
            counts.info += 1;
        }
        return counts;
    }, { errors: 0, warnings: 0, info: 0 });
}

function formatDiagnosticSummary(counts) {
    const parts = [];
    if (counts.errors > 0) {
        parts.push(`${counts.errors} error${counts.errors === 1 ? '' : 's'}`);
    }
    if (counts.warnings > 0) {
        parts.push(`${counts.warnings} warning${counts.warnings === 1 ? '' : 's'}`);
    }
    if (counts.info > 0) {
        parts.push(`${counts.info} info`);
    }
    return parts.join(' · ');
}

function renderDiagnosticMeta(diagnostic) {
    const meta = [];
    if (diagnostic.code) {
        meta.push(`<span class="diagnostic-code">${escapeHtml(diagnostic.code)}</span>`);
    }
    if (diagnostic.note) {
        meta.push(`<span class="diagnostic-note">note: ${escapeHtml(diagnostic.note)}</span>`);
    }
    if (diagnostic.help) {
        meta.push(`<span class="diagnostic-help">help: ${escapeHtml(diagnostic.help)}</span>`);
    }

    if (meta.length === 0) {
        return '';
    }

    return `<div class="diagnostic-meta">${meta.join('')}</div>`;
}

function formatDiagnosticMessage(diagnostic) {
    const parts = [diagnostic.message];
    if (diagnostic.note) {
        parts.push(`note: ${diagnostic.note}`);
    }
    if (diagnostic.help) {
        parts.push(`help: ${diagnostic.help}`);
    }
    return parts.join('\n\n');
}

function applyDiagnosticDecorations(diagnostics) {
    if (!editor || !window.monaco) {
        return;
    }

    const decorationSpecs = diagnostics.map((diagnostic) => ({
        range: new monaco.Range(
            diagnostic.line || 1,
            1,
            diagnostic.line || 1,
            1,
        ),
        options: {
            isWholeLine: true,
            className: diagnostic.severity === 'error'
                ? 'diagnostic-line-error'
                : diagnostic.severity === 'warning'
                    ? 'diagnostic-line-warning'
                    : diagnostic.severity === 'hint'
                        ? 'diagnostic-line-hint'
                        : 'diagnostic-line-info',
            overviewRuler: {
                color: diagnostic.severity === 'error'
                    ? 'rgba(255, 107, 107, 0.9)'
                    : diagnostic.severity === 'warning'
                        ? 'rgba(255, 209, 102, 0.9)'
                        : 'rgba(90, 200, 250, 0.9)',
                position: monaco.editor.OverviewRulerLane.Right,
            },
        },
    }));

    diagnosticDecorationIds = editor.deltaDecorations(
        diagnosticDecorationIds,
        decorationSpecs,
    );
}

// ============================================================
// Event listeners
// ============================================================
function setupEventListeners() {
    // Compile button
    document.getElementById('btn-compile').addEventListener('click', compile);

    // Format button
    document.getElementById('btn-format').addEventListener('click', formatSource);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            compile();
        }
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'F') {
            e.preventDefault();
            formatSource();
        }
    });

    // Example selector
    document.getElementById('select-example').addEventListener('change', (e) => {
        const key = e.target.value;
        if (key && EXAMPLES[key]) {
            editor.setValue(EXAMPLES[key]);
            e.target.value = '';
        }
    });

    // Tab switching
    document.querySelectorAll('.output-tabs .panel-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            document.querySelectorAll('.output-tabs .panel-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`tab-${target}`).classList.add('active');

            // Resize editors when tabs change
            if (target === 'wgsl' && wgslEditor) {
                wgslEditor.layout();
            }
        });
    });

    // Clear diagnostics
    document.getElementById('btn-clear-diag').addEventListener('click', () => {
        updateDiagnostics([]);
        if (window.monaco) {
            monaco.editor.setModelMarkers(editor.getModel(), 'fwgsl', []);
        }
    });
}

// ============================================================
// Resize handlers
// ============================================================
function setupResizeHandlers() {
    const gutterH = document.getElementById('gutter-h');
    const gutterV = document.getElementById('gutter-v');
    const panelEditor = document.getElementById('panel-editor');
    const panelDiag = document.getElementById('panel-diagnostics');

    let isResizingH = false;
    let isResizingV = false;

    gutterH.addEventListener('mousedown', (e) => {
        isResizingH = true;
        gutterH.classList.add('active');
        document.body.style.cursor = 'col-resize';
        e.preventDefault();
    });

    gutterV.addEventListener('mousedown', (e) => {
        isResizingV = true;
        gutterV.classList.add('active');
        document.body.style.cursor = 'row-resize';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (isResizingH) {
            const workspace = document.querySelector('.workspace');
            const rect = workspace.getBoundingClientRect();
            const ratio = (e.clientX - rect.left) / rect.width;
            const clamped = Math.max(0.2, Math.min(0.8, ratio));
            panelEditor.style.flex = `0 0 ${clamped * 100}%`;
        }
        if (isResizingV) {
            const panelRight = document.querySelector('.panel-right');
            const rect = panelRight.getBoundingClientRect();
            const diagHeight = rect.bottom - e.clientY;
            const clamped = Math.max(60, Math.min(rect.height - 120, diagHeight));
            panelDiag.style.height = `${clamped}px`;
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizingH) {
            isResizingH = false;
            gutterH.classList.remove('active');
            document.body.style.cursor = '';
        }
        if (isResizingV) {
            isResizingV = false;
            gutterV.classList.remove('active');
            document.body.style.cursor = '';
        }
    });
}

// ============================================================
// Utilities
// ============================================================
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================================
// Start
// ============================================================
init().catch(console.error);
