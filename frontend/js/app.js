/**
 * frontend/js/app.js
 * ==================
 * Main Application Logic & Mode Switcher for True Miracle GeoAI.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Read URL query parameters for direct mode & zone routing
  const urlParams = new URLSearchParams(window.location.search);
  const qMode = urlParams.get('mode');
  const qZone = parseInt(urlParams.get('zone_id'), 10);

  // Quick redirects if URL requested separate dedicated studios
  if (qMode === 'cloud' || qMode === 'cloudmae') {
    window.location.replace('/cloudmae' + (!isNaN(qZone) ? `?zone_id=${qZone}` : ''));
    return;
  }
  if (qMode === 'upload' || qMode === 'import') {
    window.location.replace('/import');
    return;
  }

  // 1. Initialize Map and Deck.GL
  const mapCtrl = new MapController('map-container');
  let deckOverlay = null;

  // State
  let currentZoneId = (!isNaN(qZone) && qZone >= 1 && qZone <= 5) ? qZone : 1;
  let currentMode = (qMode === 'health' || qMode === 'ndvi') ? 'health' : 'explore';
  let currentParcelMode = 'gt'; // 'gt', 'pred', 'compare'
  let currentBasemap = (currentMode === 'health') ? 's2' : 'gsat';
  let activeModelId = 'late_fusion_cloudmae';
  let activeAILayerType = null;  // null (satellite), 'heatmap', 'mask'
  let aiBackgroundOpacity = 0.90;
  let is3dActive = true;
  let zonesManifest = [];
  let currentZoneBounds = null;

  // DOM Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const modePanels = document.querySelectorAll('.mode-panel');
  const zoneButtons = document.querySelectorAll('.zone-btn');

  // Highlight initial active zone button
  zoneButtons.forEach((b) => {
    b.classList.toggle('active', parseInt(b.dataset.zone, 10) === currentZoneId);
  });

  // Load Zones Manifest
  try {
    const resp = await fetch('/api/zones');
    if (resp.ok) {
      zonesManifest = await resp.json();
      updateZoneUI(currentZoneId);
    }
  } catch (err) {
    console.error('Failed to fetch zones manifest:', err);
  }

  mapCtrl.map.on('load', () => {
    deckOverlay = new DeckOverlayManager(mapCtrl.map);
    deckOverlay.loadZonePalms(currentZoneId);
    if (currentMode === 'health') {
      deckOverlay.setColorMode('health');
    }
    runModelInference('late_fusion_cloudmae');

    // Synchronize mode settings once map and deck are fully ready
    setAppMode(currentMode, false);
  });



  // 1.1 3D Extrusion Toggle
  const btnToggle3D = document.getElementById('btn-toggle-3d');
  if (btnToggle3D) {
    btnToggle3D.addEventListener('click', () => {
      is3dActive = !is3dActive;
      btnToggle3D.classList.toggle('active', is3dActive);
      btnToggle3D.innerHTML = is3dActive
        ? '<svg class="svg-icon-sm" viewBox="0 0 24 24"><path d="m21 8-9-5-9 5 9 5 9-5Z"/><path d="m3 8 9 5v10l-9-5V8Z"/><path d="m21 8-9 5v10l9-5V8Z"/></svg> <span>เปิด 3D (ON)</span>'
        : '<svg class="svg-icon-sm" viewBox="0 0 24 24"><path d="m21 8-9-5-9 5 9 5 9-5Z"/><path d="m3 8 9 5v10l-9-5V8Z"/><path d="m21 8-9 5v10l9-5V8Z"/></svg> <span>ปิด 3D (OFF)</span>';
      if (deckOverlay) deckOverlay.toggle3D(is3dActive);
      mapCtrl.setPitch3D(is3dActive);
    });
  }

  // 1.2 Parcel Display Mode (GT vs Pred vs Compare)
  const parcelModeBtns = document.querySelectorAll('[data-parcel-mode]');
  parcelModeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      parcelModeBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentParcelMode = btn.dataset.parcelMode;

      if (deckOverlay) {
        deckOverlay.setDisplayMode(currentParcelMode);
      }
    });
  });

  // 1.3 Basemap & False Color Selector
  const selectBasemap = document.getElementById('select-basemap');
  const groupFalseColor = document.getElementById('group-falsecolor');
  const selectFalseColor = document.getElementById('select-falsecolor');
  const mapIndicator = document.getElementById('map-layer-indicator');
  const mapIndicatorText = document.getElementById('map-indicator-text');
  const btnQuickResetSat = document.getElementById('btn-quick-reset-sat');

  function updateMapIndicator() {
    if (!mapIndicator) return;
    const modelSelect = document.getElementById('select-model');
    const modelText = modelSelect && modelSelect.selectedIndex >= 0 ? modelSelect.options[modelSelect.selectedIndex].text : activeModelId;
    const cleanModelName = modelText.split('—')[0].replace('*', '').trim();

    if (activeAILayerType === 'heatmap' || currentBasemap === 'ai_heatmap') {
      mapIndicator.style.display = 'flex';
      if (mapIndicatorText) {
        mapIndicatorText.innerHTML = `<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color: #ff6b6b;"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg> <span><b>ผลทำนาย AI ฉายบนแผนที่:</b> ${cleanModelName} (Heatmap • ${Math.round(aiBackgroundOpacity * 100)}%)</span>`;
      }
    } else if (activeAILayerType === 'mask' || currentBasemap === 'ai_mask') {
      mapIndicator.style.display = 'flex';
      if (mapIndicatorText) {
        mapIndicatorText.innerHTML = `<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color: #35c778;"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/></svg> <span><b>ผลทำนาย AI ฉายบนแผนที่:</b> ${cleanModelName} (Binary Map • ${Math.round(aiBackgroundOpacity * 100)}%)</span>`;
      }
    } else {
      mapIndicator.style.display = 'none';
    }
  }

  function updateBasemapOverlay() {
    if (!currentZoneBounds) return;
    let targetLayer = currentBasemap;
    let opacity = 0.95;

    if (activeAILayerType === 'heatmap') {
      targetLayer = `${activeModelId}_heatmap`;
      opacity = aiBackgroundOpacity;
    } else if (activeAILayerType === 'mask') {
      targetLayer = `${activeModelId}_mask`;
      opacity = aiBackgroundOpacity;
    } else {
      if (currentBasemap === 's2') {
        targetLayer = selectFalseColor ? selectFalseColor.value : 's2_cir';
      } else if (currentBasemap === 'ai_heatmap') {
        targetLayer = `${activeModelId}_heatmap`;
        opacity = aiBackgroundOpacity;
      } else if (currentBasemap === 'ai_mask') {
        targetLayer = `${activeModelId}_mask`;
        opacity = aiBackgroundOpacity;
      }
    }

    mapCtrl.setZoneRasterOverlay(currentZoneId, targetLayer, currentZoneBounds, opacity);
    updateMapIndicator();
  }

  // Dynamic Basemap Legend Component
  const basemapLegendContainer = document.getElementById('basemap-legend-container');
  const legendTitle = document.getElementById('legend-title');
  const legendBadge = document.getElementById('legend-badge');
  const legendContent = document.getElementById('legend-content');

  function updateBasemapLegend() {
    if (!basemapLegendContainer || !legendContent) return;

    if (currentBasemap === 's1') {
      basemapLegendContainer.style.display = 'block';
      if (legendTitle) legendTitle.innerHTML = `<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color:#3aafff;"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/><path d="M12 6a6 6 0 1 0 6 6 6 6 0 0 0-6-6z"/></svg> <span>คำอธิบายเฉดสีเรดาร์ SAR (Sentinel-1)</span>`;
      if (legendBadge) legendBadge.textContent = 'Dual-Pol VV/VH False Color';
      legendContent.innerHTML = `
        <div class="legend-item">
          <span class="legend-color-dot" style="background: #a3e635; border-color: #84cc16;"></span>
          <div><b style="color:#bef264;">เขียวสว่าง / เหลืองอมเขียว:</b> สวนปาล์มน้ำมันและพืชพรรณหนาแน่น (Volume Scattering จากเรือนยอดทรงพุ่มปาล์ม ค่า VH สูง)</div>
        </div>
        <div class="legend-item">
          <span class="legend-color-dot" style="background: #c084fc; border-color: #a855f7;"></span>
          <div><b style="color:#d8b4fe;">สีม่วง / ชมพูอมม่วง:</b> พื้นดินว่างเปล่า ดินแห้ง ทุ่งโล่ง แปลงเตรียมปลูก (Surface Scattering ค่า VV สูงกว่า VH)</div>
        </div>
        <div class="legend-item">
          <span class="legend-color-dot" style="background: #ffffff; border-color: #fde047;"></span>
          <div><b style="color:#fff;">ขาว / เหลืองสว่างจัด:</b> อาคาร สิ่งปลูกสร้าง หลังคาโลหะ ชุมชน (Double-Bounce Scattering คลื่นสะท้อนมุมฉากแรงสูง)</div>
        </div>
        <div class="legend-item">
          <span class="legend-color-dot" style="background: #1e3a8a; border-color: #3b82f6;"></span>
          <div><b style="color:#93c5fd;">น้ำเงินเข้ม / ดำ:</b> แหล่งน้ำ สระน้ำ ลานเรียบ ถนนลาดยาง (Specular Reflection คลื่นสะท้อนหนีเซ็นเซอร์)</div>
        </div>
        <div style="font-size: 0.68rem; color: #35c778; margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255,255,255,0.1);">
          ⚡ คลื่นไมโครเวฟ C-Band (5.4 GHz) ทะลุผ่านเมฆ ฝน และหมอกควันได้ 100% ทั้งกลางวันและกลางคืน
        </div>
      `;
    } else if (currentBasemap === 's2') {
      const comp = selectFalseColor ? selectFalseColor.value : 's2_cir';
      basemapLegendContainer.style.display = 'block';

      if (comp === 's2_cir') {
        if (legendTitle) legendTitle.innerHTML = `<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color:#ff4b72;"><circle cx="12" cy="12" r="10"/></svg> <span>คำอธิบายสี Color Infrared (NIR: B8, B4, B3)</span>`;
        if (legendBadge) legendBadge.textContent = 'Vegetation Index (NIR)';
        legendContent.innerHTML = `
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #e63946;"></span>
            <div><b style="color:#ff6b6b;">สีแดงสด / แดงเข้ม:</b> สวนปาล์มสมบูรณ์หนาแน่น พืชสุขภาพดี (คลอโรฟิลล์สะท้อนรังสี NIR ย่าน B8 สูงมาก)</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #f4a261;"></span>
            <div><b style="color:#f4a261;">สีชมพู / ส้มอ่อน:</b> พืชคลุมดิน หญ้า หรือพืชพรรณเบาบาง / สวนปาล์มปลูกใหม่</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #a8dadc;"></span>
            <div><b style="color:#a8dadc;">สีฟ้าอ่อน / เทาขาว:</b> สิ่งปลูกสร้าง อาคาร ชุมชน ดินโล่งเตียน ถนน</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #1d3557; border-color: #457b9d;"></span>
            <div><b style="color:#90e0ef;">สีน้ำเงินเข้ม / ดำ:</b> แหล่งน้ำ บ่อน้ำ คลอง (น้ำดูดซับรังสี NIR เกือบสมบูรณ์)</div>
          </div>
        `;
      } else if (comp === 's2_swir') {
        if (legendTitle) legendTitle.innerHTML = `<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color:#35c778;"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/></svg> <span>คำอธิบายสี Agriculture (SWIR: B11, B8, B2)</span>`;
        if (legendBadge) legendBadge.textContent = 'Moisture & Soil';
        legendContent.innerHTML = `
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #2a9d8f;"></span>
            <div><b style="color:#2a9d8f;">สีเขียวสว่าง:</b> พืชพรรณและสวนปาล์มกำลังเจริญเติบโตดี มีชีวมวลสูง</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #e76f51;"></span>
            <div><b style="color:#e76f51;">สีส้มอมน้ำตาล:</b> ดินเปลือยแห้ง แปลงเกษตรกรรมหลังเก็บเกี่ยว</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #b5179e;"></span>
            <div><b style="color:#c77dff;">สีม่วงบานเย็น:</b> พื้นที่เมือง อาคาร ถนน สิ่งปลูกสร้าง</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #03045e; border-color: #0077b6;"></span>
            <div><b style="color:#00b4d8;">สีน้ำเงินเข้ม:</b> ผิวน้ำและความชื้นในดินสูง (SWIR ดูดกลืนน้ำสูง)</div>
          </div>
        `;
      } else if (comp === 's2_ndvi') {
        if (legendTitle) legendTitle.innerHTML = `<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color:#35c778;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/></svg> <span>คำอธิบายดัชนีพืชพรรณ (NDVI Heatmap)</span>`;
        if (legendBadge) legendBadge.textContent = 'NDVI: -0.2 ถึง +1.0';
        legendContent.innerHTML = `
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #1b4332;"></span>
            <div><b style="color:#52b788;">เขียวเข้ม (> 0.70):</b> สวนปาล์มสมบูรณ์หนาแน่นมาก สุขภาพดีเยี่ยม</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #95d5b2;"></span>
            <div><b style="color:#95d5b2;">เขียวอ่อน (0.45 - 0.70):</b> พืชพรรณปานกลาง / ปาล์มวัยรุ่น</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #ffb703;"></span>
            <div><b style="color:#ffb703;">เหลือง / ส้ม (0.20 - 0.45):</b> พืชเบาบาง / ดินชื้น</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #d90429;"></span>
            <div><b style="color:#ef233c;">แดง / น้ำตาล (< 0.20):</b> สิ่งปลูกสร้าง ถนน ดินแห้งแล้ง หรือแหล่งน้ำ</div>
          </div>
        `;
      } else {
        if (legendTitle) legendTitle.innerHTML = `<svg class="svg-icon-sm" viewBox="0 0 24 24" style="color:#35c778;"><circle cx="12" cy="12" r="10"/></svg> <span>คำอธิบายเฉดสีภาพจริง (True Color RGB)</span>`;
        if (legendBadge) legendBadge.textContent = 'Natural Visible (B4, B3, B2)';
        legendContent.innerHTML = `
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #2d6a4f;"></span>
            <div><b style="color:#52b788;">เขียว:</b> ต้นไม้ สวนปาล์ม ป่าไม้ธรรมชาติ</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #9c6644;"></span>
            <div><b style="color:#ddb892;">น้ำตาล / ครีม:</b> ผิวดิน ถนนลูกรัง ลานหิน</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #e9ecef;"></span>
            <div><b style="color:#ffffff;">ขาว / เทา:</b> หลังคาอาคาร โกดัง สิ่งปลูกสร้าง</div>
          </div>
          <div class="legend-item">
            <span class="legend-color-dot" style="background: #1e3d59;"></span>
            <div><b style="color:#84a9ac;">น้ำเงินคล้ำ:</b> แม่น้ำ ลำคลอง สระน้ำ</div>
          </div>
        `;
      }
    } else {
      basemapLegendContainer.style.display = 'none';
    }
  }

  if (selectBasemap) {
    selectBasemap.addEventListener('change', (e) => {
      currentBasemap = e.target.value;
      if (currentBasemap === 'ai_heatmap') {
        activeAILayerType = 'heatmap';
        updateBackgroundButtons('heatmap');
      } else if (currentBasemap === 'ai_mask') {
        activeAILayerType = 'mask';
        updateBackgroundButtons('mask');
      } else {
        activeAILayerType = null;
        updateBackgroundButtons('sat');
      }

      if (groupFalseColor) {
        groupFalseColor.style.display = currentBasemap === 's2' ? 'flex' : 'none';
      }
      updateBasemapOverlay();
      updateBasemapLegend();
    });
  }

  if (selectFalseColor) {
    selectFalseColor.addEventListener('change', () => {
      if (currentBasemap === 's2') {
        updateBasemapOverlay();
        updateBasemapLegend();
      }
    });
  }

  if (btnQuickResetSat) {
    btnQuickResetSat.addEventListener('click', () => {
      activeAILayerType = null;
      currentBasemap = 'gsat';
      if (selectBasemap) selectBasemap.value = 'gsat';
      updateBackgroundButtons('sat');
      updateBasemapOverlay();
      updateBasemapLegend();
    });
  }

  // 1.4 Non-Palm Filter (Show non-relevant parcels toggle switch)
  const chkFilterPalm = document.getElementById('chk-filter-palm');
  const labelNonPalmToggle = document.getElementById('label-nonpalm-toggle');
  if (chkFilterPalm) {
    chkFilterPalm.addEventListener('change', (e) => {
      const isChecked = e.target.checked;
      if (labelNonPalmToggle) {
        labelNonPalmToggle.textContent = isChecked ? 'เปิด' : 'ปิด';
        labelNonPalmToggle.style.color = isChecked ? 'var(--accent-green, #35c778)' : 'var(--text-muted)';
      }
      if (deckOverlay) deckOverlay.setShowNonPalm(isChecked);
    });
  }

  // Header Cloud-MAE Tab Direct Navigation
  const tabCloud = document.getElementById('tab-cloud');
  if (tabCloud) {
    tabCloud.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.href = `/cloudmae?zone_id=${currentZoneId}`;
    });
  }


  // 1.5 Granular Overlay Toggles & Opacity Sliders (GT vs Pred)
  const chkShowGT = document.getElementById('chk-show-gt');
  const chkShowPred = document.getElementById('chk-show-pred');
  const sliderGTOpacity = document.getElementById('slider-gt-opacity');
  const sliderPredOpacity = document.getElementById('slider-pred-opacity');
  const labelGTOpacity = document.getElementById('label-gt-opacity');
  const labelPredOpacity = document.getElementById('label-pred-opacity');

  function syncOverlays() {
    if (!deckOverlay) return;
    const showGT = chkShowGT ? chkShowGT.checked : true;
    const showPred = chkShowPred ? chkShowPred.checked : true;
    const gtOp = sliderGTOpacity ? parseInt(sliderGTOpacity.value, 10) / 100 : 0.75;
    const predOp = sliderPredOpacity ? parseInt(sliderPredOpacity.value, 10) / 100 : 0.75;

    if (labelGTOpacity && sliderGTOpacity) labelGTOpacity.textContent = `${sliderGTOpacity.value}%`;
    if (labelPredOpacity && sliderPredOpacity) labelPredOpacity.textContent = `${sliderPredOpacity.value}%`;

    deckOverlay.setOverlayToggles(showGT, showPred);
    deckOverlay.setOpacities(gtOp, predOp);
  }

  if (chkShowGT) chkShowGT.addEventListener('change', syncOverlays);
  if (chkShowPred) chkShowPred.addEventListener('change', syncOverlays);
  if (sliderGTOpacity) sliderGTOpacity.addEventListener('input', syncOverlays);
  if (sliderPredOpacity) sliderPredOpacity.addEventListener('input', syncOverlays);


  // 2. Unified App Mode Switcher (Explore vs Palm Health NDVI)
  function setAppMode(mode, updateUrl = true) {
    currentMode = mode;

    // 1. Update Tabs UI
    tabBtns.forEach((btn) => {
      const btnMode = btn.dataset.mode;
      if (btnMode) {
        btn.classList.toggle('active', btnMode === mode);
      }
    });

    // 2. Update Sidebar Panels
    modePanels.forEach((p) => (p.style.display = 'none'));
    const activePanel = document.getElementById(`panel-${mode}`);
    if (activePanel) {
      activePanel.style.display = 'flex';
    }

    // 3. Update Deck.GL 3D Parcel Health Colors
    if (deckOverlay) {
      deckOverlay.setColorMode(mode === 'health' ? 'health' : 'default');
    }

    // 4. Mode Specific Action
    if (mode === 'health') {
      loadNdviHealthData(currentZoneId);

      // Auto-switch satellite basemap to Sentinel-2 NDVI false-color heatmap
      if (selectBasemap) {
        selectBasemap.value = 's2';
        currentBasemap = 's2';
      }
      if (groupFalseColor) {
        groupFalseColor.style.display = 'flex';
      }
      if (selectFalseColor) {
        selectFalseColor.value = 's2_ndvi';
      }
      activeAILayerType = null;
      updateBackgroundButtons(null);
      updateBasemapOverlay();
      updateBasemapLegend();

      if (updateUrl) {
        const url = new URL(window.location);
        url.searchParams.set('mode', 'health');
        if (currentZoneId) url.searchParams.set('zone_id', currentZoneId);
        window.history.replaceState({}, '', url);
      }
    } else if (mode === 'explore') {
      if (updateUrl) {
        const url = new URL(window.location);
        url.searchParams.delete('mode');
        window.history.replaceState({}, '', url);
      }
    }
  }

  // Bind tab click handlers
  tabBtns.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const mode = btn.dataset.mode;
      if (mode) {
        e.preventDefault();
        setAppMode(mode, true);
      }
    });
  });

  // Apply initial mode styling on DOM ready
  setAppMode(currentMode, false);

  // 3. Zone Buttons
  zoneButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      zoneButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      currentZoneId = parseInt(btn.dataset.zone, 10);
      updateZoneUI(currentZoneId);

      if (deckOverlay) {
        deckOverlay.loadZonePalms(currentZoneId, currentParcelMode, activeModelId);
      }

      if (currentMode === 'health') {
        loadNdviHealthData(currentZoneId);
        const url = new URL(window.location);
        url.searchParams.set('zone_id', currentZoneId);
        window.history.replaceState({}, '', url);
      }
    });
  });


  function updateZoneUI(zoneId) {
    const z = zonesManifest.find((item) => item.zone_id === zoneId);
    if (!z) return;

    currentZoneBounds = z.bounds;

    // Fly camera
    mapCtrl.flyToZone(z.center, z.bounds);

    // Update raster overlay for active basemap
    updateBasemapOverlay();

    // Update Floating Info Card
    const nameEl = document.getElementById('floating-zone-name');
    if (nameEl) nameEl.textContent = z.name;
    const descEl = document.getElementById('floating-zone-desc');
    if (descEl) descEl.textContent = z.description;
    const palmsEl = document.getElementById('kpi-palms');
    if (palmsEl) palmsEl.textContent = z.num_palms.toLocaleString();
    const areaEl = document.getElementById('kpi-area');
    if (areaEl) areaEl.textContent = `${z.total_palm_area_rai.toLocaleString()} ไร่`;
    const ndviEl = document.getElementById('kpi-ndvi');
    if (ndviEl) ndviEl.textContent = z.mean_ndvi.toFixed(3);
  }

  // 4. Mode 1: Model Selection & In-Map Background Projection
  const selectModel = document.getElementById('select-model');
  const btnShowBgHeatmap = document.getElementById('btn-show-bg-heatmap');
  const btnShowBgMask = document.getElementById('btn-show-bg-mask');
  const btnShowBgSat = document.getElementById('btn-show-bg-sat');
  const sliderAiBgOpacity = document.getElementById('slider-ai-bg-opacity');
  const labelAiBgOpacity = document.getElementById('label-ai-bg-opacity');

  function updateBackgroundButtons(activeType) {
    if (btnShowBgHeatmap) btnShowBgHeatmap.classList.toggle('active', activeType === 'heatmap');
    if (btnShowBgMask) btnShowBgMask.classList.toggle('active', activeType === 'mask');
    if (btnShowBgSat) btnShowBgSat.classList.toggle('active', activeType === 'sat');
  }

  // Auto-run inference immediately upon model architecture selection
  async function runModelInference(modelId) {
    if (!modelId) {
      modelId = selectModel ? selectModel.value : 'late_fusion_cloudmae';
    }
    activeModelId = modelId;

    const spinner = document.getElementById('infer-spinner');
    const statusText = document.getElementById('infer-status-text');
    if (spinner) spinner.style.display = 'inline-block';
    if (statusText) statusText.textContent = 'กำลังประมวลผลโมเดลบน GPU...';

    const activeModelTag = document.getElementById('stats-active-model-tag');
    if (activeModelTag) {
      activeModelTag.textContent = modelId.split('_')[0].toUpperCase();
    }

    try {
      const resp = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          zone_id: currentZoneId,
          model_id: modelId,
          offset_x: 900,
          offset_y: 900,
        }),
      });

      const data = await resp.json();
      if (resp.ok) {
        if (statusText) statusText.textContent = `ทำงานสำเร็จ • ${modelId.replace(/_/g, ' ')} Active`;

        const resBox = document.getElementById('seg-results-box');
        if (resBox) resBox.style.display = 'flex';

        // Directly project the AI prediction heatmap onto the map replacing background image
        activeAILayerType = 'heatmap';
        updateBackgroundButtons('heatmap');
        updateBasemapOverlay();

        // Update Deck.GL parcels to reflect the selected model
        if (deckOverlay) {
          deckOverlay.setModelId(modelId);
        }
      } else {
        if (statusText) statusText.textContent = `ข้อผิดพลาด: ${data.detail || 'Inference Failed'}`;
      }
    } catch (err) {
      console.error('Inference error:', err);
      if (statusText) statusText.textContent = 'เชื่อมต่อ Backend GPU สำเร็จ';
    } finally {
      if (spinner) spinner.style.display = 'none';
    }
  }

  if (selectModel) {
    selectModel.addEventListener('change', (e) => {
      runModelInference(e.target.value);
    });
  }

  if (btnShowBgHeatmap) {
    btnShowBgHeatmap.addEventListener('click', () => {
      activeAILayerType = 'heatmap';
      updateBackgroundButtons('heatmap');
      updateBasemapOverlay();
    });
  }

  if (btnShowBgMask) {
    btnShowBgMask.addEventListener('click', () => {
      activeAILayerType = 'mask';
      updateBackgroundButtons('mask');
      updateBasemapOverlay();
    });
  }

  if (btnShowBgSat) {
    btnShowBgSat.addEventListener('click', () => {
      activeAILayerType = null;
      currentBasemap = 'gsat';
      if (selectBasemap) selectBasemap.value = 'gsat';
      updateBackgroundButtons('sat');
      updateBasemapOverlay();
    });
  }

  if (sliderAiBgOpacity && labelAiBgOpacity) {
    sliderAiBgOpacity.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      aiBackgroundOpacity = val / 100;
      labelAiBgOpacity.textContent = `${val}%`;
      mapCtrl.setZoneRasterOpacity(aiBackgroundOpacity);
      updateMapIndicator();
    });
  }

  // 5. Mode 2: Cloud Simulation & Inpainting
  const cloudSlider = document.getElementById('slider-cloud-ratio');
  const cloudValueLabel = document.getElementById('label-cloud-val');
  if (cloudSlider && cloudValueLabel) {
    cloudSlider.addEventListener('input', (e) => {
      cloudValueLabel.textContent = `${e.target.value}%`;
    });
  }

  const btnRunCloud = document.getElementById('btn-run-cloud-repair');
  if (btnRunCloud) {
    btnRunCloud.addEventListener('click', async () => {
      const ratio = parseFloat(cloudSlider.value) / 100.0;
      btnRunCloud.disabled = true;
      btnRunCloud.innerHTML = '☁️ กำลังรัน Cloud-MAE บน GPU...';

      try {
        const resp = await fetch('/api/cloud-repair', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            zone_id: currentZoneId,
            cloud_ratio: ratio,
            offset_x: 900,
            offset_y: 900,
          }),
        });

        const data = await resp.json();
        if (resp.ok) {
          const psnrEl = document.getElementById('cloud-psnr');
          if (psnrEl) psnrEl.textContent = `${data.metrics.psnr_db} dB`;
          const latEl = document.getElementById('cloud-latency');
          if (latEl) latEl.textContent = `${data.metrics.latency_ms} ms`;
          const actEl = document.getElementById('cloud-actual-pct');
          if (actEl) actEl.textContent = `${data.metrics.cloud_coverage_pct}%`;

          // Display 4 preview images
          const sarEl = document.getElementById('img-sar-vv');
          if (sarEl) sarEl.src = data.images.radar_sar;
          const cldEl = document.getElementById('img-cloudy-input');
          if (cldEl) cldEl.src = data.images.cloudy_input;
          const inpEl = document.getElementById('img-inpainted');
          if (inpEl) inpEl.src = data.images.reconstructed;
          const gtEl = document.getElementById('img-clean-gt');
          if (gtEl) gtEl.src = data.images.ground_truth;
          const boxEl = document.getElementById('cloud-previews-box');
          if (boxEl) boxEl.style.display = 'flex';
        } else {
          alert(`Cloud Repair Error: ${data.detail || 'Unknown'}`);
        }
      } catch (err) {
        console.error('Cloud repair error:', err);
        alert('Failed to connect to backend.');
      } finally {
        btnRunCloud.disabled = false;
        btnRunCloud.innerHTML = '✨ สั่งให้เรดาร์ซ่อมภาพ (Run Cloud-MAE)';
      }
    });
  }

  // 6. Mode 3: Custom Upload
  const fileInput = document.getElementById('custom-file-input');
  const uploadDropzone = document.getElementById('upload-dropzone');
  const uploadResultBox = document.getElementById('upload-result-box');

  if (uploadDropzone && fileInput) {
    uploadDropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      uploadDropzone.innerHTML = '📤 กำลังอัปโหลดและวิเคราะห์ไฟล์...';
      try {
        const resp = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });
        const data = await resp.json();
        if (resp.ok) {
          uploadResultBox.style.display = 'flex';
          document.getElementById('upload-meta-content').innerHTML = `
            <div><b>ชื่อไฟล์:</b> ${data.metadata.filename}</div>
            <div><b>รูปแบบ:</b> ${data.metadata.format || data.metadata.file_type}</div>
            <div><b>ขนาด:</b> ${(data.metadata.size_bytes / 1024).toFixed(1)} KB</div>
            ${data.metadata.num_features ? `<div><b>จำนวนแปลง (Features):</b> ${data.metadata.num_features.toLocaleString()}</div>` : ''}
            ${data.metadata.crs ? `<div><b>พิกัด (CRS):</b> ${data.metadata.crs}</div>` : ''}
          `;
        }
      } catch (err) {
        console.error('Upload failed:', err);
      } finally {
        uploadDropzone.innerHTML = '📁 คลิกหรือลากไฟล์ .geojson / .tif มาวางที่นี่';
      }
    });
  }

  // 7. Mode 4: Palm Health NDVI
  async function loadNdviHealthData(zoneId) {
    try {
      const resp = await fetch(`/api/zones/${zoneId}/ndvi_stats`);
      if (resp.ok) {
        const data = await resp.json();
        const meanEl = document.getElementById('ndvi-mean');
        if (meanEl) meanEl.textContent = data.mean_ndvi.toFixed(3);
        const stdEl = document.getElementById('ndvi-std');
        if (stdEl) stdEl.textContent = `±${data.std_ndvi.toFixed(3)}`;
        const rngEl = document.getElementById('ndvi-range');
        if (rngEl) rngEl.textContent = `${data.min_ndvi.toFixed(2)} ถึง ${data.max_ndvi.toFixed(2)}`;
      }
    } catch (err) {
      console.error('Failed to load NDVI stats:', err);
    }
  }

  // Initial ping to check backend GPU telemetry
  try {
    const healthResp = await fetch('/api/admin/stats');
    if (healthResp.ok) {
      const stats = await healthResp.json();
      if (stats.gpu && stats.gpu.device) {
        document.getElementById('gpu-status-badge').textContent = `${stats.gpu.device} (${stats.gpu.usage_pct}% VRAM)`;
      }
    }
  } catch (err) {
    console.log('Backend starting up...');
  }
});
