// JS bootstrap for the MoonBit-built fwgsl wasm-gc artifact.
//
// The playground (`../app.js`) does `import('./pkg/fwgsl_wasm.js')` and
// then calls `pkg.default()` to initialise. After init the module
// itself is used as the wasm module — `pkg.compile(source)`,
// `pkg.format(source)`, `pkg.editor_completions(source, line, column)`,
// etc.
//
// MoonBit's `link.wasm-gc.use-js-builtin-string: true` enables the
// JS String Builtins proposal, which makes MoonBit `String` values
// pass to / from JS as native JS strings with no marshaling. This
// requires:
//
// - Chrome 131+
// - Firefox 133+ (behind `javascript.options.wasm_js_string_builtins`)
// - Safari 18.2+
//
// On older browsers `WebAssembly.instantiateStreaming` rejects the
// `builtins` option and instantiation fails; `app.js` falls back to
// the JS-side mock compiler.

let _initialised = false;
let _exports = null;

const WASM_URL = new URL('./fwgsl_wasm_bg.wasm', import.meta.url);

async function _instantiate() {
    const response = await fetch(WASM_URL);
    if (!response.ok) {
        throw new Error(`failed to fetch ${WASM_URL}: ${response.status}`);
    }
    // `builtins: ['js-string']` enables the JS String Builtins proposal
    // for this module so MoonBit's `String` type maps to native JS
    // strings.
    const { instance } = await WebAssembly.instantiateStreaming(
        response,
        {},
        { builtins: ['js-string'], importedStringConstants: 'moonbit:string-constants' },
    );
    return instance.exports;
}

/**
 * Initialise the wasm module. Resolves once the moonbit-emitted wasm
 * is instantiated. Subsequent calls are no-ops.
 */
export default async function init() {
    if (_initialised) {
        return;
    }
    _exports = await _instantiate();
    _initialised = true;
}

function _assertReady() {
    if (!_initialised || _exports === null) {
        throw new Error(
            'fwgsl_wasm: call `await init()` before invoking compile / format / editor_*',
        );
    }
}

/** Full parse -> semantic -> AST lowering -> MIR -> WGSL pipeline. */
export function compile(source) {
    _assertReady();
    return _exports.compile(source);
}

/**
 * Run only the front half of the pipeline (parse + semantic + lint).
 * Returns a JSON array of diagnostics.
 */
export function wasm_check(source) {
    _assertReady();
    return _exports.wasm_check(source);
}

/** Alias for `compile`. */
export function wasm_compile(source) {
    _assertReady();
    return _exports.wasm_compile(source);
}

/** Token-stream formatter. */
export function format(source) {
    _assertReady();
    return _exports.format(source);
}

/** Monaco-shaped completion list. */
export function editor_completions(source, line, column) {
    _assertReady();
    return _exports.editor_completions(source, line, column);
}

/** Monaco-shaped hover info or the string `"null"`. */
export function editor_hover(source, line, column) {
    _assertReady();
    return _exports.editor_hover(source, line, column);
}

/** Monaco-shaped goto-definition locations. */
export function editor_definition(source, line, column) {
    _assertReady();
    return _exports.editor_definition(source, line, column);
}

/**
 * Monaco-shaped reference locations. Aliased to definition until a
 * dedicated find-references walk lands.
 */
export function editor_references(source, line, column, includeDeclaration) {
    _assertReady();
    return _exports.editor_references(source, line, column, !!includeDeclaration);
}

/**
 * Feature detect for the JS String Builtins proposal. Lets the host
 * page decide whether to even attempt to load the moonbit-built wasm.
 */
export function supportsJsStringBuiltins() {
    try {
        // Minimal valid wasm header.
        const buffer = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
        WebAssembly.instantiate(buffer, {}, { builtins: ['js-string'] });
        return true;
    } catch {
        return false;
    }
}
