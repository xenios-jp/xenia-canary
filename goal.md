# Metal Shader Loading Rewrite Goal

## Objective

Make Metal shader startup and runtime misses behave like the faster D3D12 path:
guest shader discovery is shared through `.xsh`, pipeline descriptions are shared
through `.xpso`, Metal artifacts are packed in one local file, and a warm cache
never reruns Xenos-to-DXBC, DXBC-to-DXIL, or Metal Shader Converter work before
the draw can proceed.

## Apple-aligned model

- Treat `MTLLibrary` objects as runtime load units created from already-produced
  Metal IR libraries.
- Keep `MTLBinaryArchive` as the driver-side pipeline cache, seeded by real
  pipeline descriptors and serialized on shutdown or explicit flush.
- Use asynchronous work for runtime misses so the render thread skips the draw
  until the translated shader and pipeline are ready.
- Defer Metal 4 compiler API adoption until after this cache spine is correct;
  it is a better compiler-control layer, not a replacement for persisted guest
  shaders, Metal artifacts, or binary archives.

## Minimal redesign

1. Remove the legacy Metal-only cache flow:
   - No per-shader `.metalshcache` files.
   - No custom `.metal.pipelines` stream.
   - No eager startup translation from cached pipeline entries.

2. Adopt the shared shader storage spine:
   - Use `ShaderStorageWriter<MetalPipelineStoredDescription>`.
   - Persist guest ucode in the shared `{TITLE_ID}.xsh`.
   - Persist fixed-size Metal pipeline descriptions in `{TITLE_ID}.metal.xpso`.
   - Queue shader and pipeline writes from the same runtime points D3D12 uses.

3. Add a packed local Metal artifact store:
   - One append-only `{TITLE_ID}.metal.artifacts` file under local shader
     storage.
   - Records keyed by guest ucode hash, shader modification, stage, and artifact
     kind.
   - Payload stores function name, Metal library bytes, and the binding/reflection
     metadata needed to bind the shader without retranslation.
   - Invalidates old Metal cache formats by version and ABI tag.

4. Fast load path:
   - Load `.xsh`, `.metal.xpso`, artifact index, and `MTLBinaryArchive`.
   - Hydrate cached shader metadata immediately.
   - Create `MTLLibrary` / `MTLFunction` lazily from artifact bytes only when a
     draw or warmup pipeline needs them.
   - Never call Xenos-to-DXBC, DXBC-to-DXIL, or MSC on an artifact hit.

5. Runtime miss path:
   - Queue async translation and pipeline creation.
   - Skip the draw until the requested state is published.
   - Persist guest shader, Metal artifact, pipeline description, and binary
     archive entry when creation succeeds.

6. Warmup policy:
   - Blocking startup prewarms only already-stored pipelines that have all
     required artifacts.
   - Warmup is budgeted and uses the same creation queue as runtime misses.
   - Standard, geometry-emulation, and tessellation-emulation paths must all use
     the same artifact store.

## Validation

- `git diff --check`
- Metal GPU target build
- App target build when practical
- Confirm no references remain to `.metalshcache` or `.metal.pipelines`
- Confirm warm-cache shader hits do not invoke DXBC-to-DXIL or MSC conversion
