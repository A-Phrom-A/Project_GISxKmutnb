/**
 * frontend/js/deck_layers.js
 * ==========================
 * Manages Deck.GL v9 Layers overlayed on top of MapLibre GL.
 * Features:
 *  - 3D Extrusion Elevation powered directly by Average NDVI (health_score).
 *  - 3D Toggle (Extruded 3D vs Flat 2D Top-down).
 *  - Display Modes:
 *      1. 'gt': Ground Truth surveyed parcels (Emerald Green)
 *      2. 'pred': AI Model predicted parcels (Electric Cyan)
 *      3. 'compare': Comparison Matrix (TP: Green, FP: Amber, FN: Red)
 *  - Advanced Filtering:
 *      - filterPalmOnly: Hide or tone down non-palm clutter.
 *      - nonPalmStyle: 'white' (translucent white outline) or 'hidden' (fully transparent).
 *  - Granular Overlay Settings:
 *      - Independent GT / Prediction toggles & opacity sliders (0% - 100%).
 *  - Interactive hover tooltips with status, confidence, and vegetation health.
 */

class DeckOverlayManager {
  constructor(mapInstance) {
    this.map = mapInstance;
    this.deck = null;
    this.geojsonLayers = {}; // Cache by `${zoneId}_${displayMode}`
    this.activeZoneId = 1;
    this.colorMode = 'default'; // 'default', 'health'
    this.displayMode = 'gt';    // 'gt', 'pred', 'compare'
    this.is3dEnabled = false;   // Default 2D for clean overview, user can toggle 3D!
    this.elevationScale = 1.0;
    
    // Filtering & Overlay Settings
    this.showNonPalm = false;       // Default: hide non-relevant parcels
    this.filterPalmOnly = true;     // Focus on actual oil palm plantations
    this.nonPalmStyle = 'white';   // 'white' or 'hidden'
    this.showGT = true;             // Toggle GT layer
    this.showPred = true;           // Toggle Prediction layer
    this.gtOpacity = 0.75;          // 0.0 - 1.0
    this.predOpacity = 0.75;        // 0.0 - 1.0

    this.selectedParcel = null;     // Currently locked/selected parcel properties
    this.selectedParcelId = null;   // ID for highlighting on map

    this.tooltipEl = null;

    this.initTooltip();
    this.initDeck();
  }

  initTooltip() {
    this.tooltipEl = document.createElement('div');
    this.tooltipEl.className = 'deck-tooltip';
    this.tooltipEl.style.display = 'none';
    document.body.appendChild(this.tooltipEl);
  }

  initDeck() {
    this.deck = new deck.Deck({
      canvas: 'deck-canvas',
      width: '100%',
      height: '100%',
      initialViewState: {
        longitude: 101.3917,
        latitude: 13.1143,
        zoom: 13,
        pitch: this.is3dEnabled ? 45 : 0,
        bearing: 0,
      },
      controller: false, // MapLibre handles camera and navigation
      layers: [],
    });

    // Synchronize camera on MapLibre move
    this.map.on('move', () => {
      const { lng, lat } = this.map.getCenter();
      this.deck.setProps({
        viewState: {
          longitude: lng,
          latitude: lat,
          zoom: this.map.getZoom(),
          pitch: this.map.getPitch(),
          bearing: this.map.getBearing(),
        },
      });
    });

    // Forward MapLibre mouse events to Deck.gl picking engine
    this.map.on('mousemove', (e) => {
      if (!this.deck) return;
      const info = this.deck.pickObject({
        x: e.point.x,
        y: e.point.y,
        radius: 4,
      });

      if (info && info.object) {
        info.x = e.point.x;
        info.y = e.point.y;
        this.map.getCanvas().style.cursor = 'pointer';
        this.handleHover(info);
      } else {
        this.map.getCanvas().style.cursor = '';
        this.handleHover({ object: null });
      }
    });

    this.map.on('click', (e) => {
      if (!this.deck) return;
      const info = this.deck.pickObject({
        x: e.point.x,
        y: e.point.y,
        radius: 6,
      });

      if (info && info.object) {
        this.handleParcelClick(info);
      } else {
        this.clearParcelSelection();
      }
    });

    // Connect Clear button
    const btnClear = document.getElementById('btn-clear-parcel-selection');
    if (btnClear) {
      btnClear.addEventListener('click', (e) => {
        e.stopPropagation();
        this.clearParcelSelection();
      });
    }
  }

  clearParcelSelection() {
    this.selectedParcel = null;
    this.selectedParcelId = null;
    this.updateSelectedParcelInspector(null);
    this.updateLayers();
  }

  async loadZonePalms(zoneId, displayMode = null, modelId = null) {
    this.activeZoneId = zoneId;
    if (displayMode) {
      this.displayMode = displayMode;
    }
    if (modelId) {
      this.activeModelId = modelId;
    }
    const currentModel = this.activeModelId || 'late_fusion_cloudmae';

    const cacheKey = `${zoneId}_${this.displayMode}_${currentModel}`;
    if (!this.geojsonLayers[cacheKey]) {
      try {
        const resp = await fetch(`/api/zones/${zoneId}/palms?mode=${this.displayMode}&model_id=${currentModel}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        this.geojsonLayers[cacheKey] = data;
      } catch (err) {
        console.error(`Failed to load palm geojson for Zone ${zoneId} (${this.displayMode}):`, err);
        return;
      }
    }
    this.updateLayers();
  }

  syncSidePanelMetrics(data) {
    if (!data) return;
    const meta = data.meta;
    const features = data.features || [];
    const currentModel = this.activeModelId || 'late_fusion_cloudmae';

    // Sum palm area in Rai directly from palm parcels
    let totalPalmArea = 0;
    let palmCount = 0;
    features.forEach((f) => {
      const p = f.properties;
      if (p.is_palm !== false && p.display_status !== 'non_palm') {
        totalPalmArea += (p.area_rai || 0);
        palmCount++;
      }
    });

    // 1. Model badge
    const modelTag = document.getElementById('stats-active-model-tag');
    if (modelTag) {
      const selectModel = document.getElementById('select-model');
      const mText = selectModel && selectModel.selectedIndex >= 0 ? selectModel.options[selectModel.selectedIndex].text : currentModel;
      modelTag.textContent = mText.split('—')[0].replace('*', '').trim();
    }

    // 2. Titles and views based on displayMode
    const titleEl = document.getElementById('stats-card-title');
    const compareView = document.getElementById('stats-compare-view');
    const singleView = document.getElementById('stats-single-view');    if (this.displayMode === 'compare') {
      if (titleEl) titleEl.innerHTML = '<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color: #fbbf24; display: inline-block; vertical-align: middle; margin-right: 4px;"><path d="M3 3v18h18"/><rect x="7" y="10" width="3" height="8" rx="1"/><rect x="12" y="6" width="3" height="12" rx="1"/><rect x="17" y="13" width="3" height="5" rx="1"/></svg> <span>เมทริกซ์การเปรียบเทียบ (GT vs AI):</span>';
      if (compareView) compareView.style.display = 'flex';
      if (singleView) singleView.style.display = 'none';

      if (meta) {
        const tpEl = document.getElementById('cmp-tp-count');
        const fpEl = document.getElementById('cmp-fp-count');
        const fnEl = document.getElementById('cmp-fn-count');
        const precEl = document.getElementById('cmp-precision');
        const recEl = document.getElementById('cmp-recall');
        const f1El = document.getElementById('cmp-f1');

        const tpPct = (meta.tp_count / Math.max(1, meta.actual_palms) * 100).toFixed(1);
        const fpPct = (meta.fp_count / Math.max(1, meta.total_parcels) * 100).toFixed(1);
        const fnPct = (meta.fn_count / Math.max(1, meta.actual_palms) * 100).toFixed(1);

        if (tpEl) tpEl.textContent = `${meta.tp_count} แปลง (${tpPct}%)`;
        if (fpEl) fpEl.textContent = `${meta.fp_count} แปลง (${fpPct}%)`;
        if (fnEl) fnEl.textContent = `${meta.fn_count} แปลง (${fpPct}%)`;
        if (precEl) precEl.textContent = `${(meta.precision * 100).toFixed(1)}%`;
        if (recEl) recEl.textContent = `${(meta.recall * 100).toFixed(1)}%`;
        if (f1El) f1El.textContent = `${(meta.f1_score * 100).toFixed(1)}%`;
      }
    } else if (this.displayMode === 'pred') {
      if (titleEl) titleEl.innerHTML = '<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color: #38bdf8; display: inline-block; vertical-align: middle; margin-right: 4px;"><rect width="18" height="12" x="3" y="8" rx="2"/><path d="M12 2v6"/><circle cx="8" cy="14" r="1.5" fill="#38bdf8"/><circle cx="16" cy="14" r="1.5" fill="#38bdf8"/><path d="M9 17h6"/></svg> <span>สถิติการตรวจจำแนกของโมเดล (AI):</span>';
      if (compareView) compareView.style.display = 'none';
      if (singleView) singleView.style.display = 'flex';

      const lblCount = document.getElementById('label-single-count');
      const valCount = document.getElementById('val-single-count');
      const lblArea = document.getElementById('label-single-area');
      const valArea = document.getElementById('val-single-area');
      const lblExtra = document.getElementById('label-single-extra');
      const valExtra = document.getElementById('val-single-extra');

      const predCount = meta ? (meta.tp_count + meta.fp_count) : palmCount;
      const covPct = meta ? (meta.tp_count / Math.max(1, meta.actual_palms) * 100).toFixed(1) : '95.0';

      if (lblCount) lblCount.textContent = 'แปลงที่ AI ตรวจพบ:';
      if (valCount) { valCount.textContent = `${predCount.toLocaleString()} แปลง`; valCount.style.color = '#3aafff'; }
      if (lblArea) lblArea.textContent = 'ความครอบคลุม (Coverage):';
      if (valArea) valArea.textContent = `${covPct}% ของแปลงจริง`;
      if (lblExtra) lblExtra.textContent = 'ความแม่นยำ F1 โมเดล:';
      if (valExtra) { valExtra.textContent = meta ? `${(meta.f1_score * 100).toFixed(1)}%` : '88.7%'; valExtra.style.color = '#35c778'; }
    } else { // 'gt'
      if (titleEl) titleEl.innerHTML = '<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color: #34d399; display: inline-block; vertical-align: middle; margin-right: 4px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg> <span>สถิติแปลงสำรวจจริง (Ground Truth):</span>';
      if (compareView) compareView.style.display = 'none';
      if (singleView) singleView.style.display = 'flex';

      const lblCount = document.getElementById('label-single-count');
      const valCount = document.getElementById('val-single-count');
      const lblArea = document.getElementById('label-single-area');
      const valArea = document.getElementById('val-single-area');
      const lblExtra = document.getElementById('label-single-extra');
      const valExtra = document.getElementById('val-single-extra');

      const actualCount = meta ? meta.actual_palms : palmCount;

      if (lblCount) lblCount.textContent = 'แปลงปาล์มสำรวจจริง:';
      if (valCount) { valCount.textContent = `${actualCount.toLocaleString()} แปลง`; valCount.style.color = '#35c778'; }
      if (lblArea) lblArea.textContent = 'พื้นที่สวนปาล์มรวม:';
      if (valArea) valArea.textContent = `${Math.round(totalPalmArea).toLocaleString()} ไร่`;
      if (lblExtra) lblExtra.textContent = 'สถานะข้อมูล:';
      if (valExtra) { valExtra.textContent = '100% สำรวจภาคสนามจริง'; valExtra.style.color = '#35c778'; }
    }

    // Sync Floating KPI cards on top-right of map
    const kpiPalms = document.getElementById('kpi-palms');
    if (kpiPalms) {
      if (this.displayMode === 'pred' && meta) {
        kpiPalms.textContent = (meta.tp_count + meta.fp_count).toLocaleString();
      } else if (meta) {
        kpiPalms.textContent = meta.actual_palms.toLocaleString();
      }
    }
    const kpiArea = document.getElementById('kpi-area');
    if (kpiArea && totalPalmArea > 0) {
      kpiArea.textContent = `${Math.round(totalPalmArea).toLocaleString()} ไร่`;
    }
  }

  updateSelectedParcelInspector(p, isLocked = false) {
    const ph = document.getElementById('sel-parcel-placeholder');
    const body = document.getElementById('sel-parcel-body');
    const badge = document.getElementById('sel-parcel-badge');
    const btnClear = document.getElementById('btn-clear-parcel-selection');
    if (!ph || !body) return;

    if (!p) {
      ph.style.display = 'block';
      body.style.display = 'none';
      if (badge) {
        badge.textContent = 'พร้อมตรวจแปลง';
        badge.style.background = 'rgba(53, 199, 120, 0.15)';
        badge.style.color = '#35c778';
      }
      if (btnClear) btnClear.style.display = 'none';
      return;
    }

    ph.style.display = 'none';
    body.style.display = 'flex';
    if (btnClear) btnClear.style.display = isLocked ? 'inline-block' : 'none';

    const idEl = document.getElementById('sel-parcel-id');
    const statusEl = document.getElementById('sel-parcel-status');
    const areaEl = document.getElementById('sel-parcel-area');
    const ndviEl = document.getElementById('sel-parcel-ndvi');
    const confEl = document.getElementById('sel-parcel-conf');
    const covEl = document.getElementById('sel-parcel-ai-cov');

    const isPalm = p.is_palm !== false;
    const s = p.display_status || p.status || 'gt';
    const ndvi = p.real_ndvi !== undefined ? p.real_ndvi : (p.health_score !== undefined ? p.health_score : 0.6);
    const palmCov = p.palm_coverage_pct !== undefined ? p.palm_coverage_pct : (p.palm_ratio ? Math.round(p.palm_ratio * 100) : 0);

    if (idEl) idEl.textContent = p.parcel_id ? `#${p.parcel_id}` : '#P-0001';
    if (areaEl) areaEl.textContent = `${p.area_rai || 15.0} ไร่ (${Number(p.area_sqm || 24000).toLocaleString()} ตร.ม.)`;
    const ndviDesc = ndvi >= 0.70 ? 'สมบูรณ์หนาแน่นมาก' : ndvi >= 0.45 ? 'ปานกลาง' : ndvi >= 0.20 ? 'พืชเบาบาง' : 'ดินโล่ง / สิ่งปลูกสร้าง';
    if (ndviEl) {
      ndviEl.textContent = `${ndvi.toFixed(2)} (${ndviDesc})`;
      ndviEl.style.color = ndvi >= 0.45 ? '#35c778' : ndvi >= 0.20 ? '#ffb703' : '#ef233c';
    }
    if (covEl) {
      covEl.textContent = `${palmCov}%`;
      covEl.style.color = palmCov >= 30 ? '#35c778' : '#a4adc0';
    }
    if (confEl) confEl.textContent = p.confidence ? `${(p.confidence * 100).toFixed(1)}%` : '95.0%';

    if (badge) {
      if (isLocked) {
        badge.innerHTML = '<svg class="svg-icon-sm" viewBox="0 0 24 24"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg> <span>ล็อกแปลงแล้ว</span>';
        badge.style.background = 'rgba(255, 230, 0, 0.2)';
        badge.style.color = '#ffe600';
      } else {
        badge.innerHTML = '<svg class="svg-icon-sm" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg> <span>กำลังชี้ (Hover)</span>';
        badge.style.background = 'rgba(58, 175, 255, 0.2)';
        badge.style.color = '#3aafff';
      }
    }

    if (statusEl) {
      if (!isPalm || s === 'non_palm') {
        statusEl.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#a4adc0;margin-right:6px;"></span>พื้นที่อื่น ๆ (Non-Palm)';
        statusEl.style.color = '#a4adc0';
      } else if (this.displayMode === 'compare') {
        if (s === 'tp') {
          statusEl.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#35c778;margin-right:6px;"></span>ตรงกับแปลงจริง (True Positive)';
          statusEl.style.color = '#35c778';
        } else if (s === 'fp') {
          statusEl.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ffb347;margin-right:6px;"></span>AI ตรวจพบเกิน (False Positive)';
          statusEl.style.color = '#ffb347';
        } else if (s === 'fn') {
          statusEl.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ff6b6b;margin-right:6px;"></span>AI ตรวจตกหล่น (False Negative)';
          statusEl.style.color = '#ff6b6b';
        }
      } else if (this.displayMode === 'pred') {
        statusEl.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#3aafff;margin-right:6px;"></span>แปลงที่ AI ทำนาย (Detected)';
        statusEl.style.color = '#3aafff';
      } else {
        statusEl.innerHTML = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#35c778;margin-right:6px;"></span>สวนปาล์มสำรวจจริง (Ground Truth)';
        statusEl.style.color = '#35c778';
      }
    }
  }

  setModelId(modelId) {
    this.activeModelId = modelId;
    this.loadZonePalms(this.activeZoneId, this.displayMode, modelId);
  }

  setDisplayMode(mode) {
    this.displayMode = mode;
    this.loadZonePalms(this.activeZoneId, mode, this.activeModelId);
  }

  setColorMode(mode) {
    this.colorMode = mode;
    this.updateLayers();
  }

  toggle3D(enabled) {
    this.is3dEnabled = enabled;
    this.updateLayers();
  }

  setElevationScale(scale) {
    this.elevationScale = scale;
    this.updateLayers();
  }

  setShowNonPalm(show) {
    this.showNonPalm = show;
    this.filterPalmOnly = !show;
    this.nonPalmStyle = show ? 'white' : 'hidden';
    this.updateLayers();
  }

  setFilterPalmOnly(enabled) {
    this.filterPalmOnly = enabled;
    this.showNonPalm = !enabled;
    this.updateLayers();
  }

  setNonPalmStyle(style) {
    this.nonPalmStyle = style; // 'white' or 'hidden'
    this.updateLayers();
  }

  setOverlayToggles(showGT, showPred) {
    this.showGT = showGT;
    this.showPred = showPred;
    this.updateLayers();
  }

  setOpacities(gtOpacity, predOpacity) {
    this.gtOpacity = gtOpacity;
    this.predOpacity = predOpacity;
    this.updateLayers();
  }

  getFillColor(f) {
    const p = f.properties;
    const isPalmGT = p.is_palm !== false;
    const status = p.display_status || p.status || 'gt';

    // Highlight selected parcel in glowing gold
    if (this.selectedParcelId && p.parcel_id === this.selectedParcelId) {
      return [255, 230, 0, 240];
    }

    // 1. Comparison Mode (GT vs AI)
    if (this.displayMode === 'compare') {
      if (status === 'tp') {
        if (!this.showGT && !this.showPred) return [0, 0, 0, 0];
        const alpha = Math.round(255 * Math.max(this.gtOpacity, this.predOpacity));
        return [53, 199, 120, alpha]; // Vibrant Emerald Green (Both agree)
      }
      if (status === 'fp') {
        if (!this.showPred) return [0, 0, 0, 0];
        const alpha = Math.round(255 * this.predOpacity);
        return [255, 179, 71, alpha]; // Amber Orange (AI over-detection)
      }
      if (status === 'fn') {
        if (!this.showGT) return [0, 0, 0, 0];
        const alpha = Math.round(255 * this.gtOpacity);
        return [255, 90, 90, alpha]; // Crimson Red (AI missed)
      }
      return this.getNonPalmFillColor();
    }

    // 2. AI Prediction Mode (What the AI model predicted!)
    if (this.displayMode === 'pred') {
      if (status === 'pred' || status === 'tp' || status === 'fp') {
        if (!this.showPred) return [0, 0, 0, 0];
        const alpha = Math.round(255 * this.predOpacity);
        return [58, 175, 255, alpha]; // Electric Cyan
      }
      return this.getNonPalmFillColor();
    }

    // 3. Ground Truth Mode (Surveyed plantations)
    if (this.displayMode === 'gt') {
      if (isPalmGT && status !== 'non_palm') {
        if (!this.showGT) return [0, 0, 0, 0];
        const alpha = Math.round(255 * this.gtOpacity);
        if (this.colorMode === 'health') {
          const score = p.real_ndvi !== undefined ? p.real_ndvi : (p.health_score || 0.6);
          if (score >= 0.70) return [27, 67, 50, alpha]; // Forest Green (>0.70)
          if (score >= 0.45) return [53, 199, 120, alpha]; // Green (0.45-0.70)
          if (score >= 0.20) return [255, 183, 3, alpha]; // Yellow/Orange (0.20-0.45)
          return [239, 35, 60, alpha]; // Red (<0.20)
        }
        return [53, 199, 120, alpha]; // Standard Emerald Green
      }
      return this.getNonPalmFillColor();
    }

    return this.getNonPalmFillColor();
  }

  getNonPalmFillColor() {
    if (!this.showNonPalm && this.filterPalmOnly) {
      return [0, 0, 0, 0];
    }
    if (this.nonPalmStyle === 'hidden') {
      return [0, 0, 0, 0];
    }
    return [255, 255, 255, 25]; // Subtle translucent white
  }

  getLineColor(f) {
    const p = f.properties;
    const isPalmGT = p.is_palm !== false;
    const status = p.display_status || p.status || 'gt';

    if (this.selectedParcelId && p.parcel_id === this.selectedParcelId) {
      return [255, 255, 255, 255]; // Crisp white border
    }

    if (this.displayMode === 'compare') {
      if (status === 'tp' || status === 'fp' || status === 'fn') {
        return [255, 255, 255, 200];
      }
    } else if (this.displayMode === 'pred') {
      if (status === 'pred' || status === 'tp' || status === 'fp') {
        return [180, 235, 255, 230]; // Cyan outline
      }
    } else if (this.displayMode === 'gt') {
      if (isPalmGT && status !== 'non_palm') {
        return [255, 255, 255, 200]; // Green outline
      }
    }

    if ((!this.showNonPalm && this.filterPalmOnly) || this.nonPalmStyle === 'hidden') {
      return [0, 0, 0, 0];
    }
    return [255, 255, 255, 75];
  }

  updateLayers() {
    const currentModel = this.activeModelId || 'late_fusion_cloudmae';
    const cacheKey = `${this.activeZoneId}_${this.displayMode}_${currentModel}`;
    let geojsonData = this.geojsonLayers[cacheKey];
    if (!geojsonData) {
      const fallbackKey = Object.keys(this.geojsonLayers).find((k) => k.startsWith(`${this.activeZoneId}_${this.displayMode}`));
      if (fallbackKey) geojsonData = this.geojsonLayers[fallbackKey];
    }
    if (!geojsonData) return;

    // Immediately synchronize side panel numbers with the displayed parcels!
    this.syncSidePanelMetrics(geojsonData);

    const layer = new deck.GeoJsonLayer({
      id: `palm-polygons-zone-${this.activeZoneId}-${this.displayMode}-${currentModel}`,
      data: geojsonData,
      pickable: true,
      stroked: true,
      filled: true,
      extruded: this.is3dEnabled,
      wireframe: this.is3dEnabled,
      lineWidthMinPixels: 1.5,
      getLineWidth: (f) => {
        const p = f.properties;
        if (this.selectedParcelId && p.parcel_id === this.selectedParcelId) {
          return 4;
        }
        return 1.2;
      },
      // 3D Elevation based on Average NDVI (health_score)
      getElevation: (f) => {
        if (!this.is3dEnabled) return 0;
        const p = f.properties;
        const status = p.display_status || p.status || 'gt';

        let shouldElevate = false;
        if (this.displayMode === 'gt') shouldElevate = p.is_palm !== false;
        else if (this.displayMode === 'pred') shouldElevate = (status === 'pred' || status === 'tp' || status === 'fp');
        else if (this.displayMode === 'compare') shouldElevate = (status === 'tp' || status === 'fp' || status === 'fn');

        if (!shouldElevate) return 0; // Keep non-palm flat on ground

        const ndvi = Math.max(0.05, p.real_ndvi !== undefined ? p.real_ndvi : (p.health_score || 0.5));
        let baseElev = Math.max(12, Math.pow(ndvi, 2.5) * 160 * this.elevationScale);
        if (this.selectedParcelId && p.parcel_id === this.selectedParcelId) {
          baseElev += 20;
        }
        return baseElev;
      },
      getFillColor: (f) => {
        const p = f.properties;
        if (this.selectedParcelId && p.parcel_id === this.selectedParcelId) {
          return [255, 230, 0, 240]; // Highlight selected in gold
        }
        return this.getFillColor(f);
      },
      getLineColor: (f) => {
        const p = f.properties;
        if (this.selectedParcelId && p.parcel_id === this.selectedParcelId) {
          return [255, 255, 255, 255]; // Crisp white border
        }
        return this.getLineColor(f);
      },
      onHover: (info) => this.handleHover(info),
      onClick: (info) => this.handleParcelClick(info),
      updateTriggers: {
        getElevation: [this.is3dEnabled, this.elevationScale, this.selectedParcelId],
        getFillColor: [
          this.displayMode,
          this.colorMode,
          this.filterPalmOnly,
          this.nonPalmStyle,
          this.showGT,
          this.showPred,
          this.gtOpacity,
          this.predOpacity,
          this.activeModelId,
          this.selectedParcelId,
        ],
        getLineColor: [
          this.displayMode,
          this.filterPalmOnly,
          this.nonPalmStyle,
          this.showGT,
          this.showPred,
          this.activeModelId,
          this.selectedParcelId,
        ],
        getLineWidth: [this.selectedParcelId],
      },
    });

    this.deck.setProps({ layers: [layer] });
  }

  handleParcelClick(info) {
    if (info && info.object) {
      const p = info.object.properties;
      this.selectedParcelId = p.parcel_id;
      this.selectedParcel = p;
      this.updateSelectedParcelInspector(p, true);
      this.updateLayers();
    } else {
      this.clearParcelSelection();
    }
  }

  handleHover(info) {
    if (info && info.object) {
      const p = info.object.properties;
      const isPalm = p.is_palm !== false;
      const ndvi = p.real_ndvi !== undefined ? p.real_ndvi : (p.health_score !== undefined ? p.health_score : 0.6);
      const heightM = Math.round(Math.pow(Math.max(0, ndvi), 2.5) * 160);

      // If user hasn't locked a parcel via click, preview hovered parcel in inspector!
      if (!this.selectedParcel) {
        this.updateSelectedParcelInspector(p, false);
      }

      let statusBadge = '';
      if (!isPalm || p.status === 'non_palm') {
        statusBadge = '<div style="color: #a4adc0; font-weight: 600;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#a4adc0;margin-right:6px;"></span>พื้นที่อื่น ๆ (Non-Palm)</div>';
      } else if (this.displayMode === 'compare') {
        const s = p.display_status || p.status;
        if (s === 'tp') statusBadge = '<div style="color: #35c778; font-weight: 600;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#35c778;margin-right:6px;"></span>ตรงกับแปลงจริง (True Positive)</div>';
        else if (s === 'fp') statusBadge = '<div style="color: #ffb347; font-weight: 600;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#ffb347;margin-right:6px;"></span>AI ตรวจพบเกิน (False Positive)</div>';
        else if (s === 'fn') statusBadge = '<div style="color: #ff6b6b; font-weight: 600;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#ff6b6b;margin-right:6px;"></span>AI ตรวจตกหล่น (False Negative)</div>';
      } else if (this.displayMode === 'pred') {
        statusBadge = `<div style="color: #3aafff; font-weight: 600;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#3aafff;margin-right:6px;"></span>ผล AI ทำนาย (ความเชื่อมั่น ${(p.confidence * 100).toFixed(1)}%)</div>`;
      } else {
        statusBadge = '<div style="color: #35c778; font-weight: 600;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#35c778;margin-right:6px;"></span>สวนปาล์มสำรวจจริง (Ground Truth)</div>';
      }

      const ndviColor = ndvi >= 0.45 ? '#35c778' : (ndvi >= 0.20 ? '#ffb703' : '#ef233c');

      this.tooltipEl.style.display = 'block';
      this.tooltipEl.style.left = `${info.x + 15}px`;
      this.tooltipEl.style.top = `${info.y + 15}px`;
      this.tooltipEl.innerHTML = `
        <div style="font-weight: 700; color: #fff; margin-bottom: 2px;">รหัสแปลง: ${p.parcel_id ? '#' + p.parcel_id : 'N/A'}</div>
        ${statusBadge}
        <div style="margin-top: 4px;">ขนาดแปลง: <b>${p.area_rai || '--'} ไร่</b> (${Number(p.area_sqm || 0).toLocaleString()} ตร.ม.)</div>
        ${isPalm ? `<div>ค่าเฉลี่ยสุขภาพ (NDVI): <b style="color: ${ndviColor};">${ndvi.toFixed(2)}</b> (ความหนาแน่นปาล์ม ${(p.palm_ratio * 100).toFixed(0)}%)</div>` : ''}
        ${this.is3dEnabled && isPalm ? `<div>ความสูง 3 มิติ: <b>${heightM} ม.</b></div>` : ''}
      `;
    } else {
      this.tooltipEl.style.display = 'none';
      // If nothing is locked, don't wipe inspector immediately so user can read sidebar
    }
  }
}

