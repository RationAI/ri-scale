/**
 * ?overlay=<url> (repeatable): import GeoJSON overlays (level-0 pixel coordinates) into the annotation layer of the open slide.
 * Feature styling: properties.presetID if present, else "lymph_node" for class lymph_node, else "tissue_<class>".
 * The first overlay replaces the existing annotations, the following ones are added.
 */
addPlugin("overlay_url", class extends XOpatPlugin {
    constructor(id) {
        super(id);
    }

    async pluginReady() {
        const params = new URLSearchParams(window.location.search);
        const optional = new Set(params.getAll("overlay_opt")); // like overlay, but a missing file is not an error
        const urls = [...params.getAll("overlay"), ...optional];
        if (!urls.length) return;
        const viewer = await this._whenSlideOpen();
        await new Promise(r => setTimeout(r, 1200)); // let the annotation canvas initialise
        let first = true, loaded = 0, objects = 0;
        const problems = [];
        for (const url of urls) {
            try {
                const response = await fetch(url);
                if (response.status === 404 && optional.has(url)) continue;
                if (!response.ok) throw new Error("HTTP " + response.status);
                const data = await response.json();
                for (const f of data.features || []) {
                    const p = f.properties = f.properties || {};
                    if (!p.presetID && p.class) p.presetID = p.class === "lymph_node" ? "lymph_node" : "tissue_" + p.class;
                }
                const fabric = OSDAnnotations.instance().getFabric(viewer);
                const ok = await fabric.import(JSON.stringify(data), {format: "geo-json", filename: url.split("/").pop()}, first);
                if (ok) { loaded++; objects += (data.features || []).length; first = false; }
                else problems.push(url.split("/").pop() + ": nothing imported");
            } catch (e) {
                console.error("overlay failed:", url, e);
                problems.push(url.split("/").pop() + ": " + (e && e.message || e));
            }
        }
        // thin outlines: thousands of tile regions with the default width look like a black grid
        try {
            const width = Number.parseFloat(params.get("stroke") ?? "1");
            OSDAnnotations.instance().setAnnotationCommonVisualProperty("originalStrokeWidth", width);
        } catch (e) { console.warn("overlay: could not set the outline width", e); }
        window.__overlayUrlResult = {loaded, objects, problems};
        Dialogs.show(problems.length ? "Overlay problems: " + problems.join("; ") : `Overlay loaded: ${objects} objects from ${loaded} file(s).`,
            problems.length ? 7000 : 3000, problems.length ? Dialogs.MSG_WARN : Dialogs.MSG_INFO);
    }

    _whenSlideOpen() {
        return new Promise(resolve => {
            const timer = setInterval(() => {
                if (typeof VIEWER !== "undefined" && VIEWER?.world?.getItemCount() > 0) { clearInterval(timer); resolve(VIEWER); }
            }, 300);
        });
    }
});
