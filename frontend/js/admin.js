/**
 * frontend/js/admin.js
 * ====================
 * Admin Dashboard & Audit Trail Visualizer.
 */

document.addEventListener('DOMContentLoaded', () => {
  const btnRefresh = document.getElementById('btn-refresh-logs');
  const tableBody = document.getElementById('logs-table-body');

  async function loadAdminData() {
    try {
      // 1. Fetch Stats & GPU Telemetry
      const statsResp = await fetch('/api/admin/stats');
      if (statsResp.ok) {
        const stats = await statsResp.json();
        document.getElementById('stat-total-reqs').textContent = stats.summary.total_requests.toLocaleString();
        document.getElementById('stat-avg-latency').textContent = `${stats.summary.avg_latency_ms} ms`;
        document.getElementById('stat-unique-ips').textContent = stats.summary.unique_clients;
        
        if (stats.gpu) {
          document.getElementById('stat-gpu-name').textContent = stats.gpu.device || 'N/A';
          document.getElementById('stat-gpu-vram').textContent = `${stats.gpu.vram_allocated_mb} / ${stats.gpu.vram_total_mb} MB (${stats.gpu.usage_pct}%)`;
        }

        // Render popular models
        const modelsListEl = document.getElementById('popular-models-list');
        if (modelsListEl && stats.summary.popular_models) {
          modelsListEl.innerHTML = stats.summary.popular_models.map(m => `
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border);">
              <span style="color: var(--text-h); font-weight: 500;">${m.model_id}</span>
              <span style="color: var(--accent-green); font-weight: 600;">${m.count} ครั้ง</span>
            </div>
          `).join('') || '<div style="color: var(--text-muted);">ยังไม่มีประวัติการเรียกโมเดล</div>';
        }
      }

      // 2. Fetch Logs Trail
      const logsResp = await fetch('/api/admin/logs?limit=50');
      if (logsResp.ok) {
        const logs = await logsResp.json();
        if (logs.length === 0) {
          tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">ยังไม่มีประวัติการใช้งาน</td></tr>`;
          return;
        }

        tableBody.innerHTML = logs.map(l => `
          <tr>
            <td><span style="font-family: monospace; font-size: 0.75rem;">${l.timestamp}</span></td>
            <td><span style="font-family: monospace; color: var(--accent-blue);">${l.client_ip}</span></td>
            <td><span class="action-tag action-${l.action}">${l.action}</span></td>
            <td>${l.zone_id ? `Zone ${l.zone_id}` : '-'}</td>
            <td>${l.model_id || '-'}</td>
            <td><b>${l.latency_ms ? `${l.latency_ms} ms` : '-'}</b></td>
            <td style="color: var(--text-muted); font-size: 0.75rem;">${l.details || '-'}</td>
          </tr>
        `).join('');
      }
    } catch (err) {
      console.error('Failed to load admin telemetry:', err);
    }
  }

  if (btnRefresh) {
    btnRefresh.addEventListener('click', loadAdminData);
  }

  // Initial load + poll every 5 seconds
  loadAdminData();
  setInterval(loadAdminData, 5000);
});
