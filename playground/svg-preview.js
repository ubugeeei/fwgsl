'use strict';

(function () {
    const SVG_IMPORT_PATTERN = /^\s*use\s+svg::asset\(\s*(['"])([^'"]+\.svg(?:\?[^'"]*)?)\1\s*\)\s+as\s+([A-Za-z_]\w*)\s*;?\s*$/;
    const SVG_DIRECTIVE_PATTERN = /^\s*--\s*@svg\s+(.+?)\s*$/;
    const SVG_STAGE_BLOCK_PATTERN = /^\s*svg\s+stage\s*\{\s*$/;
    const SVG_LAYER_BLOCK_PATTERN = /^\s*svg\s+layer\s+([A-Za-z_]\w*)\s*\{\s*$/;
    const SVG_EFFECT_BLOCK_PATTERN = /^\s*svg\s+effect\s+([A-Za-z_]\w*)\s+for\s+([A-Za-z_]\w*)\s*\{\s*$/;
    const DIRECTIVE_TOKEN_PATTERN = /"([^"\\]|\\.)*"|'([^'\\]|\\.)*'|[^\s]+/g;
    const HEX_COLOR_PATTERN = /^#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/;
    const DEFAULT_STAGE = Object.freeze({
        width: 960,
        height: 600,
        background: '#030816',
        background2: '#102246',
        accent: '#7dd3fc',
        duration: 4200,
        loop: true,
    });
    const SUPPORTED_BLEND_MODES = new Set([
        'source-over',
        'lighter',
        'screen',
        'multiply',
        'overlay',
        'soft-light',
        'hard-light',
        'color-dodge',
        'color-burn',
        'darken',
        'lighten',
    ]);
    const NAMED_EASINGS = {
        linear: [0.0, 0.0, 1.0, 1.0],
        ease: [0.25, 0.1, 0.25, 1.0],
        'ease-in': [0.42, 0.0, 1.0, 1.0],
        'ease-out': [0.0, 0.0, 0.58, 1.0],
        'ease-in-out': [0.42, 0.0, 0.58, 1.0],
    };
    const FUNCTIONAL_EFFECT_KINDS = new Set(['fade', 'color', 'transform', 'particles', 'glass']);
    const assetCache = new Map();

    function preprocessSource(source) {
        const lines = source.split(/\r?\n/);
        const diagnostics = [];
        const imports = [];
        const seenAliases = new Set();
        const consumedLineNumbers = new Set();

        lines.forEach((line, index) => {
            const lineNumber = index + 1;
            const importSpec = parseSvgImport(line, lineNumber);
            if (!importSpec) {
                return;
            }

            if (seenAliases.has(importSpec.alias)) {
                diagnostics.push(createDiagnostic(
                    'error',
                    `Duplicate SVG binding: ${importSpec.alias}`,
                    lineNumber,
                    'Use a unique alias for each imported SVG asset.',
                ));
            } else {
                seenAliases.add(importSpec.alias);
                imports.push(importSpec);
            }
            consumedLineNumbers.add(lineNumber);
        });

        const sceneParse = parseSvgScene(lines, imports, diagnostics);
        for (const lineNumber of sceneParse.consumedLineNumbers) {
            consumedLineNumbers.add(lineNumber);
        }

        const compilerLines = lines.map((line, index) => (
            consumedLineNumbers.has(index + 1) ? '' : line
        ));
        return {
            compilerSource: compilerLines.join('\n'),
            svgScene: sceneParse.scene,
            diagnostics,
        };
    }

    function parseSvgImport(line, lineNumber) {
        const match = line.match(SVG_IMPORT_PATTERN);
        if (!match) {
            return null;
        }

        return {
            alias: match[3],
            path: match[2],
            lineNumber,
        };
    }

    function parseSvgScene(lines, imports, diagnostics) {
        const functionalScene = parseFunctionalScene(lines, imports, diagnostics);
        let sawSceneSyntax = false;
        const stage = functionalScene.stage ? { ...functionalScene.stage } : { ...DEFAULT_STAGE };
        const layers = [...functionalScene.layers];
        const effects = [...functionalScene.effects];
        const consumedLineNumbers = new Set(functionalScene.consumedLineNumbers);

        for (let index = 0; index < lines.length; index += 1) {
            const line = lines[index];
            const lineNumber = index + 1;
            if (consumedLineNumbers.has(lineNumber)) {
                sawSceneSyntax = true;
                continue;
            }
            const directiveMatch = line.match(SVG_DIRECTIVE_PATTERN);
            if (!directiveMatch) {
                const block = parseSceneBlock(lines, index, diagnostics);
                if (!block) {
                    continue;
                }

                sawSceneSyntax = true;
                index = block.endIndex;
                for (const consumed of block.lineNumbers) {
                    consumedLineNumbers.add(consumed);
                }

                if (block.kind === 'stage') {
                    applyStageDirective(stage, block.options, lineNumber, diagnostics);
                    continue;
                }
                if (block.kind === 'layer') {
                    applyLayerDirective(
                        layers,
                        { name: block.name, options: block.options },
                        lineNumber,
                        diagnostics,
                    );
                    continue;
                }
                if (block.kind === 'effect') {
                    applyEffectDirective(
                        effects,
                        { name: block.name, target: block.target, options: block.options },
                        lineNumber,
                        diagnostics,
                    );
                }
                continue;
            }

            sawSceneSyntax = true;
            consumedLineNumbers.add(lineNumber);
            const tokens = tokenizeDirective(directiveMatch[1]);
            if (tokens.length === 0) {
                continue;
            }

            const command = tokens.shift();
            if (command === 'stage') {
                applyStageDirective(stage, parseOptions(tokens), lineNumber, diagnostics);
                continue;
            }
            if (command === 'layer') {
                applyLayerDirective(layers, parseNamedCommand(tokens), lineNumber, diagnostics);
                continue;
            }
            if (command === 'effect') {
                applyEffectDirective(effects, parseNamedCommand(tokens), lineNumber, diagnostics);
                continue;
            }

            diagnostics.push(createDiagnostic(
                'warning',
                `Unknown SVG directive: ${command}`,
                lineNumber,
                'Use `stage`, `layer`, or `effect` after `-- @svg`.',
            ));
        }

        if (!imports.length && !sawSceneSyntax) {
            return {
                scene: null,
                consumedLineNumbers,
            };
        }

        if (imports.length > 0 && layers.length === 0) {
            const centerX = stage.width / 2;
            const centerY = stage.height / 2;
            imports.forEach((asset, index) => {
                layers.push({
                    name: asset.alias,
                    asset: asset.alias,
                    x: centerX + index * 24,
                    y: centerY + index * 24,
                    width: 280,
                    height: null,
                    opacity: 1.0,
                    blend: 'source-over',
                    blur: 0,
                    color: null,
                    tint: 0,
                    translateX: 0,
                    translateY: 0,
                    scaleX: 1,
                    scaleY: 1,
                    rotate: 0,
                    lineNumber: asset.lineNumber,
                });
            });
        }

        const importMap = new Map(imports.map((item) => [item.alias, item]));
        const layerNames = new Set();
        for (const layer of layers) {
            if (layerNames.has(layer.name)) {
                diagnostics.push(createDiagnostic(
                    'error',
                    `Duplicate SVG layer name: ${layer.name}`,
                    layer.lineNumber,
                    'Rename one of the duplicated `svg layer ... { ... }` declarations.',
                ));
            } else {
                layerNames.add(layer.name);
            }

            if (!importMap.has(layer.asset)) {
                diagnostics.push(createDiagnostic(
                    'error',
                    `Unknown SVG asset binding: ${layer.asset}`,
                    layer.lineNumber,
                    'Import the asset first with `use svg::asset("...") as Alias;`.',
                ));
            }
        }

        for (const effect of effects) {
            if (!layerNames.has(effect.target)) {
                diagnostics.push(createDiagnostic(
                    'error',
                    `Unknown SVG layer target: ${effect.target}`,
                    effect.lineNumber,
                    'Point the effect at a declared `svg layer ... { ... }` block.',
                ));
            }
        }

        return {
            scene: {
                imports,
                stage,
                layers,
                effects,
            },
            consumedLineNumbers,
        };
    }

    function parseFunctionalScene(lines, imports, diagnostics) {
        const importAliases = new Set(imports.map((item) => item.alias));
        const bindings = collectTopLevelBindings(lines);
        const consumedLineNumbers = new Set();
        let stage = null;
        const layers = [];
        const effects = [];

        for (const binding of bindings) {
            const parsed = parseFunctionalBinding(binding, importAliases, diagnostics);
            if (!parsed) {
                continue;
            }

            for (const lineNumber of binding.lineNumbers) {
                consumedLineNumbers.add(lineNumber);
            }

            if (parsed.kind === 'stage') {
                if (stage) {
                    diagnostics.push(createDiagnostic(
                        'warning',
                        'Multiple functional SVG stage bindings found',
                        binding.lineNumber,
                        'The later stage binding overrides the earlier one.',
                    ));
                }
                stage = parsed.stage;
                continue;
            }

            if (parsed.kind === 'layer') {
                layers.push(parsed.layer);
                continue;
            }

            effects.push(parsed.effect);
        }

        return {
            stage,
            layers,
            effects,
            consumedLineNumbers,
        };
    }

    function collectTopLevelBindings(lines) {
        const bindings = [];

        for (let index = 0; index < lines.length; index += 1) {
            const line = lines[index];
            const match = line.match(/^(\s*)([A-Za-z_]\w*)\s*=\s*(.*)$/);
            if (!match) {
                continue;
            }

            const indent = match[1].length;
            const name = match[2];
            const expressionLines = [match[3]];
            const lineNumbers = [index + 1];
            let endIndex = index;

            for (let next = index + 1; next < lines.length; next += 1) {
                const nextLine = lines[next];
                const trimmed = nextLine.trim();

                if (!trimmed) {
                    expressionLines.push('');
                    lineNumbers.push(next + 1);
                    endIndex = next;
                    continue;
                }

                const nextIndent = countIndent(nextLine);
                if (nextIndent <= indent && /^\s*[A-Za-z_]\w*\s*=/.test(nextLine)) {
                    break;
                }
                if (nextIndent <= indent && parseSvgImport(nextLine, next + 1)) {
                    break;
                }
                if (nextIndent <= indent && (SVG_STAGE_BLOCK_PATTERN.test(nextLine) || SVG_LAYER_BLOCK_PATTERN.test(nextLine) || SVG_EFFECT_BLOCK_PATTERN.test(nextLine))) {
                    break;
                }

                expressionLines.push(nextLine.trim());
                lineNumbers.push(next + 1);
                endIndex = next;
            }

            bindings.push({
                name,
                lineNumber: index + 1,
                expression: expressionLines.join('\n').trim(),
                lineNumbers,
            });
            index = endIndex;
        }

        return bindings;
    }

    function parseFunctionalBinding(binding, importAliases, diagnostics) {
        const segments = splitTopLevelPipelines(binding.expression);
        if (!segments.length) {
            return null;
        }

        const first = parseCallSegment(segments[0]);
        if (!first) {
            return null;
        }

        const steps = segments.slice(1).map(parseCallSegment).filter(Boolean);

        if (first.name === 'canvas') {
            return {
                kind: 'stage',
                stage: parseFunctionalStage(binding, first, steps, diagnostics),
            };
        }

        if (importAliases.has(first.name) && first.args.length === 0) {
            return {
                kind: 'layer',
                layer: parseFunctionalLayer(binding, first.name, steps, diagnostics),
            };
        }

        if (FUNCTIONAL_EFFECT_KINDS.has(first.name)) {
            return {
                kind: 'effect',
                effect: parseFunctionalEffect(binding, first, steps, diagnostics),
            };
        }

        return null;
    }

    function parseFunctionalStage(binding, first, steps, diagnostics) {
        const stage = { ...DEFAULT_STAGE };
        stage.width = parsePositiveNumber(argAsNumber(first.args[0]), stage.width, binding.lineNumber, diagnostics, 'width');
        stage.height = parsePositiveNumber(argAsNumber(first.args[1]), stage.height, binding.lineNumber, diagnostics, 'height');

        for (const step of steps) {
            switch (step.name) {
                case 'background':
                    stage.background = parseColor(argAsString(step.args[0]), stage.background, binding.lineNumber, diagnostics, 'background');
                    break;
                case 'background2':
                    stage.background2 = parseColor(argAsString(step.args[0]), stage.background2, binding.lineNumber, diagnostics, 'background2');
                    break;
                case 'accent':
                    stage.accent = parseColor(argAsString(step.args[0]), stage.accent, binding.lineNumber, diagnostics, 'accent');
                    break;
                case 'duration':
                    stage.duration = parsePositiveNumber(argAsNumber(step.args[0]), stage.duration, binding.lineNumber, diagnostics, 'duration');
                    break;
                case 'loop':
                    stage.loop = parseBoolean(argAsPrimitive(step.args[0]), stage.loop);
                    break;
                default:
                    break;
            }
        }

        return stage;
    }

    function parseFunctionalLayer(binding, assetAlias, steps, diagnostics) {
        const layer = {
            name: binding.name,
            asset: assetAlias,
            x: DEFAULT_STAGE.width / 2,
            y: DEFAULT_STAGE.height / 2,
            width: null,
            height: null,
            opacity: 1.0,
            blend: 'source-over',
            blur: 0,
            color: null,
            tint: 0,
            translateX: 0,
            translateY: 0,
            scaleX: 1,
            scaleY: 1,
            rotate: 0,
            lineNumber: binding.lineNumber,
        };

        for (const step of steps) {
            switch (step.name) {
                case 'center':
                case 'at':
                    layer.x = parseNumber(argAsNumber(step.args[0]), layer.x);
                    layer.y = parseNumber(argAsNumber(step.args[1]), layer.y);
                    break;
                case 'resize':
                case 'size':
                    layer.width = parseOptionalPositiveNumber(argAsNumber(step.args[0]), layer.width, binding.lineNumber, diagnostics, 'width');
                    layer.height = parseOptionalPositiveNumber(argAsNumber(step.args[1]), layer.height, binding.lineNumber, diagnostics, 'height');
                    break;
                case 'opacity':
                case 'alpha':
                    layer.opacity = clamp(parseNumber(argAsNumber(step.args[0]), layer.opacity), 0, 1);
                    break;
                case 'blend':
                    layer.blend = parseBlendMode(argAsString(step.args[0]), layer.blend);
                    break;
                case 'blur':
                    layer.blur = Math.max(0, parseNumber(argAsNumber(step.args[0]), layer.blur));
                    break;
                case 'tint':
                    layer.color = parseOptionalColor(argAsString(step.args[0]), layer.color, binding.lineNumber, diagnostics, 'color');
                    layer.tint = clamp(parseNumber(argAsNumber(step.args[1]), 1.0), 0, 1);
                    break;
                case 'move':
                case 'translate':
                    layer.translateX = parseNumber(argAsNumber(step.args[0]), layer.translateX);
                    layer.translateY = parseNumber(argAsNumber(step.args[1]), layer.translateY);
                    break;
                case 'scale':
                    layer.scaleX = parseNumber(argAsNumber(step.args[0]), layer.scaleX);
                    layer.scaleY = parseNumber(argAsNumber(step.args[1]), layer.scaleX);
                    break;
                case 'rotate':
                    layer.rotate = parseNumber(argAsNumber(step.args[0]), layer.rotate);
                    break;
                default:
                    break;
            }
        }

        return layer;
    }

    function parseFunctionalEffect(binding, first, steps, diagnostics) {
        const effect = {
            kind: first.name,
            target: argAsIdentifier(first.args[0]) || '',
            start: 0,
            duration: 1000,
            easing: 'linear',
            lineNumber: binding.lineNumber,
        };

        if (first.name === 'fade') {
            effect.from = clamp(parseNumber(argAsNumber(first.args[1]), 0), 0, 1);
            effect.to = clamp(parseNumber(argAsNumber(first.args[2]), 1), 0, 1);
        } else if (first.name === 'color') {
            effect.from = parseColor(argAsString(first.args[1]), '#ffffff', binding.lineNumber, diagnostics, 'from');
            effect.to = parseColor(argAsString(first.args[2]), '#ffffff', binding.lineNumber, diagnostics, 'to');
            effect.mix = 1.0;
        } else if (first.name === 'transform') {
            effect.from = parseFunctionalTransformSpec(first.args[1], binding.lineNumber, diagnostics);
            effect.to = parseFunctionalTransformSpec(first.args[2], binding.lineNumber, diagnostics);
        } else if (first.name === 'particles') {
            effect.count = Math.max(8, Math.round(parseNumber(argAsNumber(first.args[1]), 160)));
            effect.spread = 180;
            effect.size = 2.4;
            effect.drift = 0.22;
        } else if (first.name === 'glass') {
            effect.shards = Math.max(4, Math.round(parseNumber(argAsNumber(first.args[1]), 16)));
            effect.spread = 220;
            effect.rotation = 1.0;
        }

        for (const step of steps) {
            switch (step.name) {
                case 'during':
                    effect.start = Math.max(0, parseNumber(argAsNumber(step.args[0]), effect.start));
                    effect.duration = Math.max(1, parseNumber(argAsNumber(step.args[1]), effect.duration));
                    break;
                case 'ease':
                    effect.easing = parseFunctionalEasingSpec(step.args[0]);
                    break;
                case 'spread':
                    effect.spread = Math.max(1, parseNumber(argAsNumber(step.args[0]), effect.spread || 1));
                    break;
                case 'size':
                    effect.size = Math.max(0.5, parseNumber(argAsNumber(step.args[0]), effect.size || 0.5));
                    break;
                case 'drift':
                    effect.drift = clamp(parseNumber(argAsNumber(step.args[0]), effect.drift || 0), -1, 1);
                    break;
                case 'rotation':
                    effect.rotation = Math.max(0, parseNumber(argAsNumber(step.args[0]), effect.rotation || 0));
                    break;
                case 'mix':
                    effect.mix = clamp(parseNumber(argAsNumber(step.args[0]), effect.mix || 1), 0, 1);
                    break;
                case 'count':
                    effect.count = Math.max(8, Math.round(parseNumber(argAsNumber(step.args[0]), effect.count || 160)));
                    break;
                case 'shards':
                    effect.shards = Math.max(4, Math.round(parseNumber(argAsNumber(step.args[0]), effect.shards || 16)));
                    break;
                default:
                    break;
            }
        }

        return effect;
    }

    function parseSceneBlock(lines, startIndex, diagnostics) {
        const line = lines[startIndex];
        const lineNumber = startIndex + 1;
        let kind = null;
        let name = null;
        let target = null;

        if (SVG_STAGE_BLOCK_PATTERN.test(line)) {
            kind = 'stage';
        } else {
            const layerMatch = line.match(SVG_LAYER_BLOCK_PATTERN);
            if (layerMatch) {
                kind = 'layer';
                name = layerMatch[1];
            }
        }

        if (!kind) {
            const effectMatch = line.match(SVG_EFFECT_BLOCK_PATTERN);
            if (effectMatch) {
                kind = 'effect';
                name = effectMatch[1];
                target = effectMatch[2];
            }
        }

        if (!kind) {
            return null;
        }

        const bodyEntries = [];
        const lineNumbers = [lineNumber];
        let endIndex = startIndex;

        for (let index = startIndex + 1; index < lines.length; index += 1) {
            const currentLineNumber = index + 1;
            const trimmed = lines[index].trim();
            lineNumbers.push(currentLineNumber);
            endIndex = index;

            if (trimmed === '}' || trimmed === '};') {
                return {
                    kind,
                    name,
                    target,
                    options: parseBlockOptions(bodyEntries, diagnostics),
                    lineNumbers,
                    endIndex,
                };
            }

            bodyEntries.push({
                text: lines[index],
                lineNumber: currentLineNumber,
            });
        }

        diagnostics.push(createDiagnostic(
            'error',
            `Unclosed SVG ${kind} block`,
            lineNumber,
            'Close the block with `}`.',
        ));

        return {
            kind,
            name,
            target,
            options: parseBlockOptions(bodyEntries, diagnostics),
            lineNumbers,
            endIndex,
        };
    }

    function tokenizeDirective(body) {
        return body.match(DIRECTIVE_TOKEN_PATTERN) || [];
    }

    function parseNamedCommand(tokens) {
        const args = [...tokens];
        const name = args.shift() || '';
        return {
            name,
            options: parseOptions(args),
        };
    }

    function parseOptions(tokens) {
        const options = {};
        for (const token of tokens) {
            const separatorIndex = token.indexOf('=');
            if (separatorIndex === -1) {
                options[token] = true;
                continue;
            }

            const key = token.slice(0, separatorIndex);
            const rawValue = token.slice(separatorIndex + 1);
            options[key] = unquoteToken(rawValue);
        }
        return options;
    }

    function parseBlockOptions(entries, diagnostics) {
        const options = {};

        for (const entry of entries) {
            const cleaned = stripInlineComment(entry.text).trim();
            if (!cleaned) {
                continue;
            }

            const match = cleaned.match(/^([A-Za-z_]\w*)\s*:\s*(.+?)\s*[,;]?\s*$/);
            if (!match) {
                diagnostics.push(createDiagnostic(
                    'warning',
                    `Invalid SVG property syntax: ${cleaned}`,
                    entry.lineNumber,
                    'Use `name: value` inside `svg stage`, `svg layer`, or `svg effect` blocks.',
                ));
                continue;
            }

            options[match[1]] = unquoteToken(match[2].trim());
        }

        return options;
    }

    function unquoteToken(token) {
        if (
            (token.startsWith('"') && token.endsWith('"'))
            || (token.startsWith('\'') && token.endsWith('\''))
        ) {
            return token.slice(1, -1).replace(/\\(["'])/g, '$1');
        }
        return token;
    }

    function stripInlineComment(line) {
        let inSingle = false;
        let inDouble = false;

        for (let index = 0; index < line.length - 1; index += 1) {
            const ch = line[index];
            const next = line[index + 1];

            if (ch === '"' && !inSingle && line[index - 1] !== '\\') {
                inDouble = !inDouble;
                continue;
            }
            if (ch === '\'' && !inDouble && line[index - 1] !== '\\') {
                inSingle = !inSingle;
                continue;
            }
            if (!inSingle && !inDouble && ch === '-' && next === '-') {
                return line.slice(0, index);
            }
        }

        return line;
    }

    function countIndent(line) {
        const match = line.match(/^\s*/);
        return match ? match[0].length : 0;
    }

    function splitTopLevelPipelines(text) {
        return splitTopLevelByOperator(text, '|>');
    }

    function splitTopLevelCompose(text) {
        const segments = [];
        let current = '';
        let depth = 0;
        let inSingle = false;
        let inDouble = false;

        for (let index = 0; index < text.length; index += 1) {
            const ch = text[index];

            if (ch === '"' && !inSingle && text[index - 1] !== '\\') {
                inDouble = !inDouble;
                current += ch;
                continue;
            }
            if (ch === '\'' && !inDouble && text[index - 1] !== '\\') {
                inSingle = !inSingle;
                current += ch;
                continue;
            }
            if (inSingle || inDouble) {
                current += ch;
                continue;
            }

            if (ch === '(' || ch === '[' || ch === '{') {
                depth += 1;
                current += ch;
                continue;
            }
            if (ch === ')' || ch === ']' || ch === '}') {
                depth = Math.max(0, depth - 1);
                current += ch;
                continue;
            }

            const prev = text[index - 1];
            const next = text[index + 1];
            if (depth === 0 && ch === '.' && /\s/.test(prev || '') && /\s/.test(next || '')) {
                if (current.trim()) {
                    segments.push(current.trim());
                }
                current = '';
                continue;
            }

            current += ch;
        }

        if (current.trim()) {
            segments.push(current.trim());
        }

        return segments;
    }

    function splitTopLevelByOperator(text, operator) {
        const segments = [];
        let current = '';
        let depth = 0;
        let inSingle = false;
        let inDouble = false;

        for (let index = 0; index < text.length; index += 1) {
            const ch = text[index];

            if (ch === '"' && !inSingle && text[index - 1] !== '\\') {
                inDouble = !inDouble;
                current += ch;
                continue;
            }
            if (ch === '\'' && !inDouble && text[index - 1] !== '\\') {
                inSingle = !inSingle;
                current += ch;
                continue;
            }
            if (inSingle || inDouble) {
                current += ch;
                continue;
            }

            if (ch === '(' || ch === '[' || ch === '{') {
                depth += 1;
                current += ch;
                continue;
            }
            if (ch === ')' || ch === ']' || ch === '}') {
                depth = Math.max(0, depth - 1);
                current += ch;
                continue;
            }

            if (depth === 0 && text.startsWith(operator, index)) {
                if (current.trim()) {
                    segments.push(current.trim());
                }
                current = '';
                index += operator.length - 1;
                continue;
            }

            current += ch;
        }

        if (current.trim()) {
            segments.push(current.trim());
        }

        return segments;
    }

    function splitTopLevelWhitespace(text) {
        const segments = [];
        let current = '';
        let depth = 0;
        let inSingle = false;
        let inDouble = false;

        for (let index = 0; index < text.length; index += 1) {
            const ch = text[index];

            if (ch === '"' && !inSingle && text[index - 1] !== '\\') {
                inDouble = !inDouble;
                current += ch;
                continue;
            }
            if (ch === '\'' && !inDouble && text[index - 1] !== '\\') {
                inSingle = !inSingle;
                current += ch;
                continue;
            }
            if (inSingle || inDouble) {
                current += ch;
                continue;
            }

            if (ch === '(' || ch === '[' || ch === '{') {
                depth += 1;
                current += ch;
                continue;
            }
            if (ch === ')' || ch === ']' || ch === '}') {
                depth = Math.max(0, depth - 1);
                current += ch;
                continue;
            }

            if (depth === 0 && /\s/.test(ch)) {
                if (current.trim()) {
                    segments.push(current.trim());
                }
                current = '';
                continue;
            }

            current += ch;
        }

        if (current.trim()) {
            segments.push(current.trim());
        }

        return segments;
    }

    function parseCallSegment(segment) {
        const tokens = splitTopLevelWhitespace(segment);
        if (!tokens.length) {
            return null;
        }

        return {
            name: tokens[0],
            args: tokens.slice(1).map(parseValueToken),
        };
    }

    function parseValueToken(token) {
        const trimmed = token.trim();
        if (!trimmed) {
            return null;
        }

        if (
            (trimmed.startsWith('"') && trimmed.endsWith('"'))
            || (trimmed.startsWith('\'') && trimmed.endsWith('\''))
        ) {
            return unquoteToken(trimmed);
        }

        if (isWrapped(trimmed, '(', ')')) {
            const inner = trimmed.slice(1, -1).trim();
            if (/^-?\d+(?:\.\d+)?$/.test(inner)) {
                return Number(inner);
            }
            return {
                type: 'expr',
                value: inner,
            };
        }

        if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) {
            return Number(trimmed);
        }

        return {
            type: 'ident',
            value: trimmed,
        };
    }

    function isWrapped(text, open, close) {
        if (!text.startsWith(open) || !text.endsWith(close)) {
            return false;
        }

        let depth = 0;
        let inSingle = false;
        let inDouble = false;
        for (let index = 0; index < text.length; index += 1) {
            const ch = text[index];
            if (ch === '"' && !inSingle && text[index - 1] !== '\\') {
                inDouble = !inDouble;
            } else if (ch === '\'' && !inDouble && text[index - 1] !== '\\') {
                inSingle = !inSingle;
            } else if (!inSingle && !inDouble) {
                if (ch === open) {
                    depth += 1;
                } else if (ch === close) {
                    depth -= 1;
                    if (depth === 0 && index !== text.length - 1) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    function argAsPrimitive(arg) {
        if (arg === null || arg === undefined) {
            return null;
        }
        if (typeof arg === 'number' || typeof arg === 'string' || typeof arg === 'boolean') {
            return arg;
        }
        if (arg.type === 'ident' || arg.type === 'expr') {
            return arg.value;
        }
        return null;
    }

    function argAsString(arg) {
        const primitive = argAsPrimitive(arg);
        return primitive === null ? '' : String(primitive);
    }

    function argAsNumber(arg) {
        if (typeof arg === 'number') {
            return arg;
        }
        const primitive = argAsPrimitive(arg);
        return primitive === null ? null : Number(primitive);
    }

    function argAsIdentifier(arg) {
        if (arg && typeof arg === 'object' && arg.type === 'ident') {
            return arg.value;
        }
        if (typeof arg === 'string') {
            return arg;
        }
        return '';
    }

    function parseFunctionalTransformSpec(arg, lineNumber, diagnostics) {
        if (!arg) {
            return '';
        }

        let expression = '';
        if (typeof arg === 'string') {
            expression = arg;
        } else if (typeof arg === 'object' && arg.type === 'expr') {
            expression = arg.value;
        } else {
            expression = argAsString(arg);
        }

        const raw = expression.trim();
        if (!raw) {
            return '';
        }

        const inner = raw.startsWith('transform(') && raw.endsWith(')')
            ? raw.slice('transform('.length, -1)
            : raw;
        const segments = splitTopLevelCompose(inner);
        const normalized = [];

        for (const segment of segments) {
            const call = parseCallSegment(segment);
            if (!call) {
                continue;
            }

            if (call.name === 'translate') {
                normalized.push(`translate(${parseNumber(argAsNumber(call.args[0]), 0)}, ${parseNumber(argAsNumber(call.args[1]), 0)})`);
                continue;
            }
            if (call.name === 'scale') {
                const sx = parseNumber(argAsNumber(call.args[0]), 1);
                const sy = parseNumber(argAsNumber(call.args[1]), sx);
                normalized.push(`scale(${sx}, ${sy})`);
                continue;
            }
            if (call.name === 'rotate') {
                normalized.push(`rotate(${parseNumber(argAsNumber(call.args[0]), 0)})`);
                continue;
            }
        }

        if (!normalized.length) {
            diagnostics.push(createDiagnostic(
                'warning',
                `Invalid transform expression: ${raw}`,
                lineNumber,
                'Use `translate x y . scale s . rotate deg`.',
            ));
        }

        return normalized.join(' ');
    }

    function parseFunctionalEasingSpec(arg) {
        if (!arg) {
            return 'linear';
        }

        if (typeof arg === 'string') {
            return normalizeIdentifier(arg);
        }
        if (typeof arg === 'object' && arg.type === 'ident') {
            return normalizeIdentifier(arg.value);
        }
        if (typeof arg === 'object' && arg.type === 'expr') {
            const call = parseCallSegment(arg.value);
            if (!call) {
                return 'linear';
            }
            if (call.name === 'cubicBezier') {
                const numbers = call.args.map(argAsNumber).map((value) => parseNumber(value, 0));
                return `cubic-bezier(${numbers[0]}, ${numbers[1]}, ${numbers[2]}, ${numbers[3]})`;
            }
            return normalizeIdentifier(call.name);
        }

        return 'linear';
    }

    function applyStageDirective(stage, options, lineNumber, diagnostics) {
        stage.width = parsePositiveNumber(options.width, stage.width, lineNumber, diagnostics, 'width');
        stage.height = parsePositiveNumber(options.height, stage.height, lineNumber, diagnostics, 'height');
        stage.background = parseColor(options.background, stage.background, lineNumber, diagnostics, 'background');
        stage.background2 = parseColor(options.background2, stage.background2, lineNumber, diagnostics, 'background2');
        stage.accent = parseColor(options.accent, stage.accent, lineNumber, diagnostics, 'accent');
        stage.duration = parsePositiveNumber(options.duration, stage.duration, lineNumber, diagnostics, 'duration');
        stage.loop = parseBoolean(options.loop, stage.loop);
    }

    function applyLayerDirective(layers, parsed, lineNumber, diagnostics) {
        if (!parsed.name || parsed.name.includes('=')) {
            diagnostics.push(createDiagnostic(
                'error',
                'SVG layer declaration needs a layer name',
                lineNumber,
                'Use `svg layer layer_name { asset: Alias ... }`.',
            ));
            return;
        }

        const options = parsed.options;
        layers.push({
            name: parsed.name,
            asset: String(options.asset || ''),
            x: parseNumber(options.x, DEFAULT_STAGE.width / 2),
            y: parseNumber(options.y, DEFAULT_STAGE.height / 2),
            width: parseOptionalPositiveNumber(options.width, null, lineNumber, diagnostics, 'width'),
            height: parseOptionalPositiveNumber(options.height, null, lineNumber, diagnostics, 'height'),
            opacity: clamp(parseNumber(options.opacity, 1.0), 0, 1),
            blend: parseBlendMode(options.blend, 'source-over'),
            blur: Math.max(0, parseNumber(options.blur, 0)),
            color: parseOptionalColor(options.color, null, lineNumber, diagnostics, 'color'),
            tint: clamp(parseNumber(options.tint, options.color ? 1.0 : 0.0), 0, 1),
            translateX: parseNumber(options.translateX, 0),
            translateY: parseNumber(options.translateY, 0),
            scaleX: parseNumber(options.scaleX, parseNumber(options.scale, 1)),
            scaleY: parseNumber(options.scaleY, parseNumber(options.scale, 1)),
            rotate: parseNumber(options.rotate, 0),
            lineNumber,
        });
    }

    function applyEffectDirective(effects, parsed, lineNumber, diagnostics) {
        if (!parsed.name || parsed.name.includes('=')) {
            diagnostics.push(createDiagnostic(
                'error',
                'SVG effect declaration needs an effect kind',
                lineNumber,
                'Use `svg effect fade for layer_name { ... }` or another built-in effect.',
            ));
            return;
        }

        const kind = parsed.name;
        if (!['fade', 'color', 'transform', 'particles', 'glass'].includes(kind)) {
            diagnostics.push(createDiagnostic(
                'error',
                `Unsupported SVG effect: ${kind}`,
                lineNumber,
                'Supported effects are `fade`, `color`, `transform`, `particles`, and `glass`.',
            ));
            return;
        }

        const options = parsed.options;
        const baseEffect = {
            kind,
            target: String(parsed.target || options.target || ''),
            start: Math.max(0, parseNumber(options.start, 0)),
            duration: Math.max(1, parseNumber(options.duration, 1000)),
            easing: String(options.easing || 'linear'),
            lineNumber,
        };

        if (kind === 'fade') {
            effects.push({
                ...baseEffect,
                from: clamp(parseNumber(options.from, 0), 0, 1),
                to: clamp(parseNumber(options.to, 1), 0, 1),
            });
            return;
        }

        if (kind === 'color') {
            effects.push({
                ...baseEffect,
                from: parseColor(options.from, '#ffffff', lineNumber, diagnostics, 'from'),
                to: parseColor(options.to, '#ffffff', lineNumber, diagnostics, 'to'),
                mix: clamp(parseNumber(options.mix, 1), 0, 1),
            });
            return;
        }

        if (kind === 'transform') {
            effects.push({
                ...baseEffect,
                from: String(options.from || ''),
                to: String(options.to || ''),
            });
            return;
        }

        if (kind === 'particles') {
            effects.push({
                ...baseEffect,
                count: Math.max(8, Math.round(parseNumber(options.count, 160))),
                spread: Math.max(1, parseNumber(options.spread, 180)),
                size: Math.max(0.5, parseNumber(options.size, 2.4)),
                drift: clamp(parseNumber(options.drift, 0.22), -1, 1),
            });
            return;
        }

        effects.push({
            ...baseEffect,
            shards: Math.max(4, Math.round(parseNumber(options.shards, 16))),
            spread: Math.max(1, parseNumber(options.spread, 220)),
            rotation: Math.max(0, parseNumber(options.rotation, 1.0)),
        });
    }

    function parseNumber(value, fallback) {
        if (value === undefined || value === null || value === '') {
            return fallback;
        }
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    }

    function parsePositiveNumber(value, fallback, lineNumber, diagnostics, label) {
        if (value === undefined || value === null || value === '') {
            return fallback;
        }
        const parsed = parseNumber(value, fallback);
        if (parsed > 0) {
            return parsed;
        }
        diagnostics.push(createDiagnostic(
            'warning',
            `Expected a positive ${label} in SVG directive`,
            lineNumber,
            `Falling back to ${fallback}.`,
        ));
        return fallback;
    }

    function parseOptionalPositiveNumber(value, fallback, lineNumber, diagnostics, label) {
        if (value === undefined || value === null || value === '') {
            return fallback;
        }
        return parsePositiveNumber(value, fallback ?? 1, lineNumber, diagnostics, label);
    }

    function parseBoolean(value, fallback) {
        if (value === undefined || value === null) {
            return fallback;
        }
        if (value === true) {
            return true;
        }
        const normalized = String(value).toLowerCase();
        if (['1', 'true', 'yes', 'on'].includes(normalized)) {
            return true;
        }
        if (['0', 'false', 'no', 'off'].includes(normalized)) {
            return false;
        }
        return fallback;
    }

    function parseBlendMode(value, fallback) {
        if (!value) {
            return fallback;
        }
        const normalized = normalizeIdentifier(String(value));
        return SUPPORTED_BLEND_MODES.has(normalized) ? normalized : fallback;
    }

    function parseColor(value, fallback, lineNumber, diagnostics, label) {
        if (!value) {
            return fallback;
        }
        if (HEX_COLOR_PATTERN.test(value)) {
            return value;
        }
        diagnostics.push(createDiagnostic(
            'warning',
            `Invalid SVG ${label} color: ${value}`,
            lineNumber,
            `Use a hex color such as ${fallback}.`,
        ));
        return fallback;
    }

    function parseOptionalColor(value, fallback, lineNumber, diagnostics, label) {
        if (!value) {
            return fallback;
        }
        return parseColor(value, fallback || '#ffffff', lineNumber, diagnostics, label);
    }

    function buildOutput(scene) {
        if (!scene) {
            return '// No SVG preview scene';
        }

        const importLines = scene.imports.map((item) => `// use svg::asset("${item.path}") as ${item.alias};`);
        const layerLines = scene.layers.map((layer) => `// svg layer ${layer.name} { asset: ${layer.asset}, x: ${Math.round(layer.x)}, y: ${Math.round(layer.y)} }`);
        const effectLines = scene.effects.map((effect) => `// svg effect ${effect.kind} for ${effect.target} { start: ${Math.round(effect.start)}, duration: ${Math.round(effect.duration)} }`);

        return [
            '// SVG preview mode',
            '// Rust-inspired SVG declarations are preprocessed before compiling fwgsl.',
            ...importLines,
            '//',
            ...layerLines,
            ...effectLines,
        ].join('\n');
    }

    async function runPreview(scene, options) {
        const canvas = options.canvas;
        const overlay = options.overlay;
        if (!canvas) {
            throw new Error('SVG preview canvas is missing');
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('2D canvas context unavailable');
        }

        const runtime = await buildRuntime(scene);
        if (!runtime.layers.length) {
            throw new Error('No SVG layers to render');
        }

        let raf = 0;
        let stopped = false;
        const startTime = performance.now();

        if (typeof options.activate === 'function') {
            options.activate();
        }
        if (overlay) {
            overlay.style.display = 'none';
            overlay.innerHTML = '';
        }

        const render = (now) => {
            if (stopped) {
                return;
            }

            const { width, height } = options.resizeCanvasToDisplaySize(canvas);
            renderRuntime(runtime, ctx, width, height, now - startTime);
            raf = requestAnimationFrame(render);
        };

        raf = requestAnimationFrame(render);

        return {
            stop() {
                stopped = true;
                if (raf) {
                    cancelAnimationFrame(raf);
                }
            },
        };
    }

    async function buildRuntime(scene) {
        const assets = new Map();
        for (const binding of scene.imports) {
            assets.set(binding.alias, await loadSvgAsset(binding.path));
        }

        const effectsByTarget = new Map();
        for (const effect of scene.effects) {
            if (!effectsByTarget.has(effect.target)) {
                effectsByTarget.set(effect.target, []);
            }
            effectsByTarget.get(effect.target).push(normalizeEffect(effect));
        }

        const layers = [];
        for (const layer of scene.layers) {
            const asset = assets.get(layer.asset);
            if (!asset) {
                continue;
            }
            layers.push(createLayerRuntime(layer, asset, effectsByTarget.get(layer.name) || []));
        }

        const maxEffectEnd = scene.effects.reduce(
            (longest, effect) => Math.max(longest, effect.start + effect.duration),
            0,
        );

        return {
            stage: {
                ...scene.stage,
                duration: Math.max(scene.stage.duration, maxEffectEnd + 320),
            },
            layers,
        };
    }

    function normalizeEffect(effect) {
        if (effect.kind === 'color') {
            return {
                ...effect,
                fromColor: hexToRgba(effect.from),
                toColor: hexToRgba(effect.to),
                easingFn: parseEasing(effect.easing),
            };
        }
        if (effect.kind === 'transform') {
            return {
                ...effect,
                fromTransform: parseTransform(effect.from),
                toTransform: parseTransform(effect.to),
                easingFn: parseEasing(effect.easing),
            };
        }
        return {
            ...effect,
            easingFn: parseEasing(effect.easing),
        };
    }

    function createLayerRuntime(layer, asset, effects) {
        const size = resolveLayerSize(layer, asset);
        const raster = rasterizeAsset(asset.image, size.width, size.height);
        const tintCanvas = document.createElement('canvas');
        tintCanvas.width = raster.canvas.width;
        tintCanvas.height = raster.canvas.height;

        const samplePool = buildSamplePool(raster.imageData);
        const effectRuntimes = effects.map((effect, index) => {
            if (effect.kind === 'particles') {
                return {
                    effect,
                    resource: createParticleResource(samplePool, raster.canvas.width, raster.canvas.height, effect, `${layer.name}:particles:${index}`),
                };
            }
            if (effect.kind === 'glass') {
                return {
                    effect,
                    resource: createGlassResource(samplePool, raster.canvas, raster.canvas.width, raster.canvas.height, effect, `${layer.name}:glass:${index}`),
                };
            }
            return { effect, resource: null };
        });

        return {
            layer: {
                ...layer,
                width: size.width,
                height: size.height,
                colorRgba: layer.color ? hexToRgba(layer.color) : null,
            },
            raster,
            tintCanvas,
            effectRuntimes,
        };
    }

    function resolveLayerSize(layer, asset) {
        const aspect = asset.width / Math.max(asset.height, 1);
        let width = layer.width;
        let height = layer.height;

        if (!width && !height) {
            width = asset.width;
            height = asset.height;
        } else if (width && !height) {
            height = width / aspect;
        } else if (!width && height) {
            width = height * aspect;
        }

        return {
            width: Math.max(1, Math.round(width || asset.width)),
            height: Math.max(1, Math.round(height || asset.height)),
        };
    }

    function rasterizeAsset(image, width, height) {
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(width));
        canvas.height = Math.max(1, Math.round(height));
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        return {
            canvas,
            imageData: ctx.getImageData(0, 0, canvas.width, canvas.height),
        };
    }

    function buildSamplePool(imageData) {
        const pool = [];
        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;
        const stride = Math.max(1, Math.floor(Math.sqrt((width * height) / 1800)));

        for (let y = 0; y < height; y += stride) {
            for (let x = 0; x < width; x += stride) {
                const index = (y * width + x) * 4;
                const alpha = data[index + 3];
                if (alpha < 18) {
                    continue;
                }

                pool.push({
                    x,
                    y,
                    color: [
                        data[index + 0],
                        data[index + 1],
                        data[index + 2],
                        alpha / 255,
                    ],
                });
            }
        }

        if (!pool.length) {
            pool.push({
                x: width / 2,
                y: height / 2,
                color: [255, 255, 255, 1],
            });
        }

        return pool;
    }

    function createParticleResource(samplePool, width, height, effect, seedLabel) {
        const random = createRandom(seedLabel);
        const particles = [];

        for (let index = 0; index < effect.count; index += 1) {
            const sample = samplePool[Math.floor(random() * samplePool.length)];
            const localX = sample.x - width / 2;
            const localY = sample.y - height / 2;
            const direction = normalizeVector({
                x: localX / Math.max(width, 1) + (random() - 0.5) * 0.45,
                y: localY / Math.max(height, 1) + (random() - 0.5) * 0.35 - effect.drift,
            });

            particles.push({
                x: localX,
                y: localY,
                color: sample.color,
                size: effect.size * (0.55 + random() * 0.95),
                velocityX: direction.x,
                velocityY: direction.y,
                lift: 0.4 + random() * 0.9,
                wobble: (random() - 0.5) * 18,
                alpha: 0.5 + random() * 0.5,
            });
        }

        return particles;
    }

    function createGlassResource(samplePool, rasterCanvas, width, height, effect, seedLabel) {
        const random = createRandom(seedLabel);
        const shards = [];

        for (let index = 0; index < effect.shards; index += 1) {
            const sample = samplePool[Math.floor(random() * samplePool.length)];
            const pointCount = 3 + Math.floor(random() * 3);
            const radius = 10 + random() * 20;
            const startAngle = random() * Math.PI * 2;
            const points = [];

            for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
                const angle = startAngle + (pointIndex / pointCount) * Math.PI * 2 + (random() - 0.5) * 0.3;
                const distance = radius * (0.55 + random() * 0.65);
                points.push({
                    x: clamp(sample.x + Math.cos(angle) * distance, 0, width - 1),
                    y: clamp(sample.y + Math.sin(angle) * distance, 0, height - 1),
                });
            }

            const bounds = polygonBounds(points);
            if (bounds.width < 3 || bounds.height < 3) {
                continue;
            }

            const patchCanvas = document.createElement('canvas');
            patchCanvas.width = Math.ceil(bounds.width) + 4;
            patchCanvas.height = Math.ceil(bounds.height) + 4;
            const patchCtx = patchCanvas.getContext('2d');
            const shiftX = 2 - bounds.minX;
            const shiftY = 2 - bounds.minY;

            patchCtx.save();
            patchCtx.translate(shiftX, shiftY);
            tracePolygon(patchCtx, points);
            patchCtx.clip();
            patchCtx.drawImage(rasterCanvas, 0, 0);
            patchCtx.restore();

            patchCtx.save();
            patchCtx.translate(shiftX, shiftY);
            patchCtx.lineWidth = 1.2;
            patchCtx.strokeStyle = 'rgba(255, 255, 255, 0.24)';
            tracePolygon(patchCtx, points);
            patchCtx.stroke();
            patchCtx.restore();

            const centroid = polygonCentroid(points);
            const localX = centroid.x - width / 2;
            const localY = centroid.y - height / 2;
            const direction = normalizeVector({
                x: localX / Math.max(width, 1) + (random() - 0.5) * 0.5,
                y: localY / Math.max(height, 1) + (random() - 0.5) * 0.4 - 0.08,
            });

            shards.push({
                canvas: patchCanvas,
                anchorX: centroid.x - bounds.minX + 2,
                anchorY: centroid.y - bounds.minY + 2,
                x: localX,
                y: localY,
                velocityX: direction.x,
                velocityY: direction.y,
                rotationSpeed: (random() * 2 - 1) * effect.rotation * 42,
                alpha: 0.48 + random() * 0.42,
            });
        }

        return shards;
    }

    async function loadSvgAsset(path) {
        const resolvedUrl = new URL(path, window.location.href).toString();
        if (assetCache.has(resolvedUrl)) {
            return assetCache.get(resolvedUrl);
        }

        const promise = (async () => {
            const response = await fetch(resolvedUrl);
            if (!response.ok) {
                throw new Error(`Failed to load ${path} (${response.status})`);
            }

            const text = await response.text();
            const size = parseSvgDimensions(text);
            const image = await svgTextToImage(text);

            return {
                path,
                resolvedUrl,
                width: size.width,
                height: size.height,
                image,
            };
        })();

        assetCache.set(resolvedUrl, promise);
        return promise;
    }

    function parseSvgDimensions(svgText) {
        try {
            const doc = new DOMParser().parseFromString(svgText, 'image/svg+xml');
            const root = doc.documentElement;
            const viewBox = root.getAttribute('viewBox');
            if (viewBox) {
                const parts = viewBox.split(/[\s,]+/).map(Number);
                if (parts.length === 4 && parts.every(Number.isFinite)) {
                    return {
                        width: Math.max(1, parts[2]),
                        height: Math.max(1, parts[3]),
                    };
                }
            }

            const width = parseFloat(root.getAttribute('width'));
            const height = parseFloat(root.getAttribute('height'));
            if (Number.isFinite(width) && Number.isFinite(height)) {
                return { width, height };
            }
        } catch (_error) {
            // Fall back to defaults below.
        }

        return { width: 256, height: 256 };
    }

    function svgTextToImage(svgText) {
        return new Promise((resolve, reject) => {
            const blob = new Blob([svgText], { type: 'image/svg+xml' });
            const url = URL.createObjectURL(blob);
            const image = new Image();
            image.onload = () => {
                URL.revokeObjectURL(url);
                resolve(image);
            };
            image.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('Failed to decode SVG image'));
            };
            image.src = url;
        });
    }

    function renderRuntime(runtime, ctx, canvasWidth, canvasHeight, elapsedMs) {
        const stageTime = runtime.stage.loop
            ? elapsedMs % runtime.stage.duration
            : Math.min(elapsedMs, runtime.stage.duration);

        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        renderBackdrop(runtime.stage, ctx, canvasWidth, canvasHeight, stageTime);

        const scale = Math.min(canvasWidth / runtime.stage.width, canvasHeight / runtime.stage.height);
        const offsetX = (canvasWidth - runtime.stage.width * scale) / 2;
        const offsetY = (canvasHeight - runtime.stage.height * scale) / 2;
        ctx.translate(offsetX, offsetY);
        ctx.scale(scale, scale);

        for (const layerRuntime of runtime.layers) {
            renderLayer(ctx, layerRuntime, stageTime);
        }

        ctx.restore();
    }

    function renderBackdrop(stage, ctx, width, height, timeMs) {
        const vertical = ctx.createLinearGradient(0, 0, 0, height);
        vertical.addColorStop(0, stage.background2);
        vertical.addColorStop(1, stage.background);
        ctx.fillStyle = vertical;
        ctx.fillRect(0, 0, width, height);

        const glowX = width * (0.64 + Math.sin(timeMs * 0.00037) * 0.05);
        const glowY = height * (0.2 + Math.cos(timeMs * 0.00029) * 0.04);
        const glow = ctx.createRadialGradient(glowX, glowY, 0, glowX, glowY, Math.max(width, height) * 0.52);
        glow.addColorStop(0, withAlpha(stage.accent, 0.24));
        glow.addColorStop(0.55, withAlpha(stage.accent, 0.08));
        glow.addColorStop(1, withAlpha(stage.accent, 0.0));
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, width, height);

        ctx.save();
        ctx.strokeStyle = withAlpha(stage.accent, 0.08);
        ctx.lineWidth = 1;
        const gridStep = Math.max(28, Math.floor(width / 18));
        for (let x = 0; x <= width; x += gridStep) {
            ctx.beginPath();
            ctx.moveTo(x + 0.5, 0);
            ctx.lineTo(x + 0.5, height);
            ctx.stroke();
        }
        for (let y = 0; y <= height; y += gridStep) {
            ctx.beginPath();
            ctx.moveTo(0, y + 0.5);
            ctx.lineTo(width, y + 0.5);
            ctx.stroke();
        }
        ctx.restore();
    }

    function renderLayer(ctx, layerRuntime, timeMs) {
        const state = evaluateLayerState(layerRuntime, timeMs);
        const baseOpacity = state.opacity * state.baseVisibility;

        if (baseOpacity > 0.002) {
            drawLayerRaster(ctx, layerRuntime, state, baseOpacity);
        }

        for (const glassState of state.glass) {
            drawGlassShards(ctx, layerRuntime, state, glassState);
        }

        for (const particleState of state.particles) {
            drawParticles(ctx, layerRuntime, state, particleState);
        }
    }

    function evaluateLayerState(layerRuntime, timeMs) {
        const layer = layerRuntime.layer;
        const state = {
            opacity: layer.opacity,
            baseVisibility: 1,
            tint: layer.tint,
            tintColor: layer.colorRgba,
            transform: {
                x: layer.translateX,
                y: layer.translateY,
                scaleX: layer.scaleX,
                scaleY: layer.scaleY,
                rotation: layer.rotate,
            },
            particles: [],
            glass: [],
        };

        for (const effectRuntime of layerRuntime.effectRuntimes) {
            const progress = effectProgress(effectRuntime.effect, timeMs);
            if (progress === null) {
                continue;
            }

            if (effectRuntime.effect.kind === 'fade') {
                state.opacity *= lerp(effectRuntime.effect.from, effectRuntime.effect.to, progress);
                continue;
            }

            if (effectRuntime.effect.kind === 'color') {
                state.tintColor = lerpColor(effectRuntime.effect.fromColor, effectRuntime.effect.toColor, progress);
                state.tint = Math.max(state.tint, effectRuntime.effect.mix);
                continue;
            }

            if (effectRuntime.effect.kind === 'transform') {
                const transform = lerpTransform(effectRuntime.effect.fromTransform, effectRuntime.effect.toTransform, progress);
                state.transform.x += transform.x;
                state.transform.y += transform.y;
                state.transform.scaleX *= transform.scaleX;
                state.transform.scaleY *= transform.scaleY;
                state.transform.rotation += transform.rotation;
                continue;
            }

            if (effectRuntime.effect.kind === 'particles') {
                state.baseVisibility *= 1 - progress * 0.84;
                state.particles.push({
                    progress,
                    effectRuntime,
                });
                continue;
            }

            state.baseVisibility *= 1 - progress * 0.72;
            state.glass.push({
                progress,
                effectRuntime,
            });
        }

        return state;
    }

    function effectProgress(effect, timeMs) {
        if (timeMs < effect.start) {
            return null;
        }
        const raw = clamp((timeMs - effect.start) / Math.max(effect.duration, 1), 0, 1);
        return effect.easingFn(raw);
    }

    function drawLayerRaster(ctx, layerRuntime, state, opacity) {
        const source = state.tintColor && state.tint > 0.001
            ? tintLayerRaster(layerRuntime, state.tintColor, state.tint)
            : layerRuntime.raster.canvas;

        ctx.save();
        ctx.globalCompositeOperation = layerRuntime.layer.blend;
        ctx.globalAlpha = clamp(opacity, 0, 1);
        ctx.filter = layerRuntime.layer.blur > 0 ? `blur(${layerRuntime.layer.blur}px)` : 'none';
        applyLayerTransform(ctx, layerRuntime.layer, state.transform);
        ctx.drawImage(source, -layerRuntime.layer.width / 2, -layerRuntime.layer.height / 2);
        ctx.restore();
    }

    function tintLayerRaster(layerRuntime, color, mixAmount) {
        const canvas = layerRuntime.tintCanvas;
        const ctx = canvas.getContext('2d');
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1;
        ctx.drawImage(layerRuntime.raster.canvas, 0, 0);
        ctx.globalCompositeOperation = 'source-atop';
        ctx.globalAlpha = clamp(mixAmount, 0, 1);
        ctx.fillStyle = rgbaToCss(color, 1);
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
        return canvas;
    }

    function drawParticles(ctx, layerRuntime, state, particleState) {
        const effect = particleState.effectRuntime.effect;
        const particles = particleState.effectRuntime.resource;
        const progress = particleState.progress;

        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        ctx.globalAlpha = clamp(state.opacity * (1 - progress * 0.25), 0, 1);
        applyLayerTransform(ctx, layerRuntime.layer, state.transform);

        for (const particle of particles) {
            const wobble = Math.sin(progress * Math.PI * 2 + particle.wobble) * (effect.spread * 0.06);
            const px = particle.x + particle.velocityX * effect.spread * progress + wobble;
            const py = particle.y + particle.velocityY * effect.spread * progress - particle.lift * effect.spread * progress * progress;
            const size = particle.size * (0.8 + progress * 0.9);
            const alpha = particle.alpha * (1 - progress);
            const color = state.tintColor
                ? lerpColor(
                    [particle.color[0], particle.color[1], particle.color[2], particle.color[3]],
                    state.tintColor,
                    Math.min(1, state.tint * 0.88),
                )
                : particle.color;

            ctx.fillStyle = rgbaToCss(color, alpha);
            ctx.beginPath();
            ctx.arc(px, py, size, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.restore();
    }

    function drawGlassShards(ctx, layerRuntime, state, glassState) {
        const effect = glassState.effectRuntime.effect;
        const shards = glassState.effectRuntime.resource;
        const progress = glassState.progress;

        ctx.save();
        ctx.globalCompositeOperation = layerRuntime.layer.blend;
        applyLayerTransform(ctx, layerRuntime.layer, state.transform);

        for (const shard of shards) {
            const progressCurve = Math.pow(progress, 0.88);
            const px = shard.x + shard.velocityX * effect.spread * progressCurve;
            const py = shard.y + shard.velocityY * effect.spread * progressCurve - effect.spread * 0.14 * progressCurve * progressCurve;
            const rotation = shard.rotationSpeed * progressCurve * (Math.PI / 180);
            const alpha = shard.alpha * state.opacity * (1 - progress * 0.68);

            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.translate(px, py);
            ctx.rotate(rotation);
            ctx.drawImage(shard.canvas, -shard.anchorX, -shard.anchorY);
            ctx.restore();
        }

        ctx.restore();
    }

    function applyLayerTransform(ctx, layer, transform) {
        ctx.translate(layer.x + transform.x, layer.y + transform.y);
        ctx.rotate(transform.rotation * Math.PI / 180);
        ctx.scale(transform.scaleX, transform.scaleY);
    }

    function parseTransform(spec) {
        const transform = {
            x: 0,
            y: 0,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
        };
        if (!spec) {
            return transform;
        }

        const pattern = /(translate|scale|rotate)\(([^)]*)\)/g;
        for (const match of spec.matchAll(pattern)) {
            const kind = match[1];
            const args = match[2].split(/[,\s]+/).map(Number).filter(Number.isFinite);
            if (kind === 'translate') {
                transform.x += args[0] || 0;
                transform.y += args[1] || 0;
                continue;
            }
            if (kind === 'scale') {
                const sx = args[0] || 1;
                const sy = args[1] || sx;
                transform.scaleX *= sx;
                transform.scaleY *= sy;
                continue;
            }
            if (kind === 'rotate') {
                transform.rotation += args[0] || 0;
            }
        }

        return transform;
    }

    function lerpTransform(from, to, t) {
        return {
            x: lerp(from.x, to.x, t),
            y: lerp(from.y, to.y, t),
            scaleX: lerp(from.scaleX, to.scaleX, t),
            scaleY: lerp(from.scaleY, to.scaleY, t),
            rotation: lerp(from.rotation, to.rotation, t),
        };
    }

    function parseEasing(spec) {
        if (!spec) {
            return (value) => value;
        }

        const normalized = normalizeIdentifier(spec);
        const tuple = NAMED_EASINGS[normalized];
        if (tuple) {
            return cubicBezier(tuple[0], tuple[1], tuple[2], tuple[3]);
        }

        const bezierMatch = normalized.match(/^cubic-bezier\(([^)]+)\)$/);
        if (!bezierMatch) {
            return (value) => value;
        }

        const values = bezierMatch[1].split(',').map(Number).filter(Number.isFinite);
        if (values.length !== 4) {
            return (value) => value;
        }

        return cubicBezier(values[0], values[1], values[2], values[3]);
    }

    function normalizeIdentifier(value) {
        return String(value)
            .trim()
            .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
            .replace(/_/g, '-')
            .toLowerCase();
    }

    function cubicBezier(x1, y1, x2, y2) {
        if (x1 === y1 && x2 === y2) {
            return (value) => value;
        }

        const cx = 3 * x1;
        const bx = 3 * (x2 - x1) - cx;
        const ax = 1 - cx - bx;
        const cy = 3 * y1;
        const by = 3 * (y2 - y1) - cy;
        const ay = 1 - cy - by;

        const sampleCurveX = (t) => ((ax * t + bx) * t + cx) * t;
        const sampleCurveY = (t) => ((ay * t + by) * t + cy) * t;
        const sampleCurveDerivativeX = (t) => (3 * ax * t + 2 * bx) * t + cx;

        return (value) => {
            let t = clamp(value, 0, 1);

            for (let index = 0; index < 8; index += 1) {
                const x = sampleCurveX(t) - value;
                const derivative = sampleCurveDerivativeX(t);
                if (Math.abs(x) < 1e-6 || Math.abs(derivative) < 1e-6) {
                    break;
                }
                t -= x / derivative;
            }

            let start = 0;
            let end = 1;
            for (let index = 0; index < 10; index += 1) {
                const x = sampleCurveX(t);
                if (Math.abs(x - value) < 1e-5) {
                    break;
                }
                if (x > value) {
                    end = t;
                } else {
                    start = t;
                }
                t = (start + end) / 2;
            }

            return sampleCurveY(clamp(t, 0, 1));
        };
    }

    function polygonBounds(points) {
        const xs = points.map((point) => point.x);
        const ys = points.map((point) => point.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        return {
            minX,
            minY,
            width: maxX - minX,
            height: maxY - minY,
        };
    }

    function polygonCentroid(points) {
        let area = 0;
        let x = 0;
        let y = 0;

        for (let index = 0; index < points.length; index += 1) {
            const current = points[index];
            const next = points[(index + 1) % points.length];
            const cross = current.x * next.y - next.x * current.y;
            area += cross;
            x += (current.x + next.x) * cross;
            y += (current.y + next.y) * cross;
        }

        if (Math.abs(area) < 1e-6) {
            return {
                x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
                y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
            };
        }

        return {
            x: x / (3 * area),
            y: y / (3 * area),
        };
    }

    function tracePolygon(ctx, points) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let index = 1; index < points.length; index += 1) {
            ctx.lineTo(points[index].x, points[index].y);
        }
        ctx.closePath();
    }

    function createRandom(seedLabel) {
        let seed = hashString(seedLabel);
        return function next() {
            seed += 0x6D2B79F5;
            let t = seed;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    function hashString(value) {
        let hash = 2166136261;
        for (let index = 0; index < value.length; index += 1) {
            hash ^= value.charCodeAt(index);
            hash = Math.imul(hash, 16777619);
        }
        return hash >>> 0;
    }

    function normalizeVector(vector) {
        const length = Math.hypot(vector.x, vector.y) || 1;
        return {
            x: vector.x / length,
            y: vector.y / length,
        };
    }

    function hexToRgba(hex) {
        const normalized = hex.replace('#', '');
        if (normalized.length === 8) {
            return [
                parseInt(normalized.slice(0, 2), 16),
                parseInt(normalized.slice(2, 4), 16),
                parseInt(normalized.slice(4, 6), 16),
                parseInt(normalized.slice(6, 8), 16) / 255,
            ];
        }
        return [
            parseInt(normalized.slice(0, 2), 16),
            parseInt(normalized.slice(2, 4), 16),
            parseInt(normalized.slice(4, 6), 16),
            1,
        ];
    }

    function rgbaToCss(rgba, alphaOverride) {
        const alpha = alphaOverride === undefined ? rgba[3] : alphaOverride;
        return `rgba(${Math.round(rgba[0])}, ${Math.round(rgba[1])}, ${Math.round(rgba[2])}, ${clamp(alpha, 0, 1)})`;
    }

    function withAlpha(hex, alpha) {
        const rgba = hexToRgba(hex);
        return rgbaToCss(rgba, alpha);
    }

    function lerpColor(left, right, t) {
        return [
            lerp(left[0], right[0], t),
            lerp(left[1], right[1], t),
            lerp(left[2], right[2], t),
            lerp(left[3] ?? 1, right[3] ?? 1, t),
        ];
    }

    function lerp(from, to, t) {
        return from + (to - from) * clamp(t, 0, 1);
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function createDiagnostic(severity, message, lineNumber, help) {
        return {
            severity,
            message,
            help,
            line: lineNumber,
            col: 1,
            endLine: lineNumber,
            endCol: 2,
        };
    }

    window.FWGSL_SVG_PREVIEW = {
        preprocessSource,
        buildOutput,
        runPreview,
    };
})();
