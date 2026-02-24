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

// ============================================================
// App state
// ============================================================
let editor = null;
let wgslEditor = null;
let wasmModule = null;
let gpuDevice = null;

// ============================================================
// Initialize
// ============================================================
async function init() {
    await initMonaco();
    await initWasm();
    await initWebGPU();
    setupEventListeners();
    setupResizeHandlers();
}

async function initMonaco() {
    return new Promise((resolve) => {
        require.config({
            paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs' }
        });
        require(['vs/editor/editor.main'], function (monaco) {
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
function compile() {
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
        document.getElementById('editor-status').textContent = 'compiled';
        document.getElementById('status-compile-time').textContent = `${elapsed}ms`;
        document.getElementById('output-status').textContent = `${elapsed}ms`;

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

    if (!diagnostics || diagnostics.length === 0) {
        list.innerHTML = '<div class="diagnostics-empty"><span>No diagnostics</span></div>';
        badge.textContent = '0';
        badge.className = 'badge';
        return;
    }

    badge.textContent = diagnostics.length;
    const hasErrors = diagnostics.some(d => d.severity === 'error');
    badge.className = `badge ${hasErrors ? 'has-errors' : 'has-warnings'}`;

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
                <span class="diagnostic-message">${escapeHtml(d.message)}</span>
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
                    : monaco.MarkerSeverity.Info,
            startLineNumber: d.line || 1,
            startColumn: d.col || 1,
            endLineNumber: d.endLine || d.line || 1,
            endColumn: d.endCol || (d.col || 1) + 1,
            message: d.message,
        }));
        monaco.editor.setModelMarkers(model, 'fwgsl', markers);
    }
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
