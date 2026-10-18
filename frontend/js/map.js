/**
 * frontend/js/map.js
 * ==================
 * MapLibre GL Map Controller & Geospatial Navigation.
 * Supports:
 *  - Free High-Resolution Satellite Basemap (ESRI World Imagery + Labels, no API key).
 *  - Dynamic Raster Overlays (GSat, Sentinel-1 Radar, Sentinel-2 with False Color Composites).
 *  - Smooth 3D / 2D Camera Pitch Animation.
 */

class MapController {
  constructor(containerId) {
    this.containerId = containerId;
    this.map = null;
    this.currentRasterLayer = null;
    this.initMap();
  }

  initMap() {
    // High-performance, free raster satellite basemap (No API Key Required)
    const mapStyle = {
      version: 8,
      sources: {
        'esri-satellite': {
          type: 'raster',
          tiles: [
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          ],
          tileSize: 256,
          attribution: '&copy; Esri, Maxar, Earthstar Geographics',
          maxzoom: 19,
        },
        'esri-labels': {
          type: 'raster',
          tiles: [
            'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
          ],
          tileSize: 256,
          maxzoom: 19,
        },
      },
      layers: [
        {
          id: 'esri-satellite-layer',
          type: 'raster',
          source: 'esri-satellite',
          minzoom: 0,
          maxzoom: 19,
        },
        {
          id: 'esri-labels-layer',
          type: 'raster',
          source: 'esri-labels',
          minzoom: 0,
          maxzoom: 19,
          paint: {
            'raster-opacity': 0.85,
          },
        },
      ],
    };

    this.map = new maplibregl.Map({
      container: this.containerId,
      style: mapStyle,
      center: [101.3917, 13.1143],
      zoom: 13,
      pitch: 0,
      bearing: 0,
      antialias: true,
    });

    // Add navigation controls
    this.map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-left');
  }

  flyToZone(center, bounds) {
    if (bounds) {
      this.map.fitBounds(
        [
          [bounds[0], bounds[1]],
          [bounds[2], bounds[3]],
        ],
        { padding: 60, duration: 1800, pitch: this.map.getPitch() }
      );
    } else if (center) {
      this.map.flyTo({
        center: center,
        zoom: 13.5,
        pitch: this.map.getPitch(),
        speed: 1.2,
        curve: 1.42,
        essential: true,
      });
    }
  }

  setPitch3D(is3d) {
    if (!this.map) return;
    this.map.easeTo({
      pitch: is3d ? 48 : 0,
      duration: 800,
    });
  }

  setZoneRasterOverlay(zoneId, layerType, bounds, opacity = 0.95) {
    if (!this.map) return;

    this.currentRasterLayer = layerType;
    const sourceId = 'zone-raster-source';
    const layerId = 'zone-raster-layer';

    // Remove existing layer and source if present
    if (this.map.getLayer(layerId)) {
      this.map.removeLayer(layerId);
    }
    if (this.map.getSource(sourceId)) {
      this.map.removeSource(sourceId);
    }

    if (!layerType || layerType === 'none' || layerType === 'gsat_global') {
      return; // Fall back to base satellite imagery
    }

    if (!bounds || bounds.length !== 4) return;

    // Image coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    const coords = [
      [bounds[0], bounds[3]], // Top-Left
      [bounds[2], bounds[3]], // Top-Right
      [bounds[2], bounds[1]], // Bottom-Right
      [bounds[0], bounds[1]], // Bottom-Left
    ];

    const url = `/api/zones/${zoneId}/layer/${layerType}`;

    this.map.addSource(sourceId, {
      type: 'image',
      url: url,
      coordinates: coords,
    });

    // Insert beneath labels so city/road names remain legible
    const beforeLayer = this.map.getLayer('esri-labels-layer') ? 'esri-labels-layer' : undefined;
    this.map.addLayer(
      {
        id: layerId,
        type: 'raster',
        source: sourceId,
        paint: {
          'raster-opacity': opacity,
          'raster-fade-duration': 250,
        },
      },
      beforeLayer
    );
  }

  setZoneRasterOpacity(opacity) {
    if (!this.map) return;
    const layerId = 'zone-raster-layer';
    if (this.map.getLayer(layerId)) {
      this.map.setPaintProperty(layerId, 'raster-opacity', opacity);
    }
  }
}

