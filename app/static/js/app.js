/**
 * Crawlbot (کرال‌بات) - Frontend Application Script
 * Vanilla JavaScript implementation with async API integration, state management,
 * polling, real-time filtering, URL parsing, and file export.
 */

'use strict';

// ============================================================================
// 1. State Management
// ============================================================================
const state = {
    activeTab: 'discover',
    discoveryJobId: null,
    extractionJobId: null,
    discoveredPages: [],
    selectedUrls: new Set(),
    extractionUrls: new Set(),
    extractedDocuments: [],
    isDiscovering: false,
    isExtracting: false,
    pollInterval: null,
    activePollingType: null, // 'discovery' | 'extraction'
    searchQuery: ''
};

// ============================================================================
// 2. API Client
// ============================================================================
const api = {
    /**
     * Start Discovery job
     * @param {Object} config 
     * @returns {Promise<Object>}
     */
    async startDiscovery(config) {
        const response = await fetch('/api/discovery/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.message || `خطا در شروع کشف صفحات (${response.status})`);
        }
        const data = await response.json();
        if (data && data.data && data.data.job_id && !data.job_id) {
            data.job_id = data.data.job_id;
        }
        return data;
    },

    /**
     * Get Discovery job status
     * @param {string} jobId 
     * @returns {Promise<Object>}
     */
    async getDiscoveryStatus(jobId) {
        const response = await fetch(`/api/discovery/status/${encodeURIComponent(jobId)}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.message || `خطا در دریافت وضعیت کشف (${response.status})`);
        }
        return await response.json();
    },

    /**
     * Cancel Discovery job
     * @param {string} jobId 
     * @returns {Promise<Object>}
     */
    async cancelDiscovery(jobId) {
        const response = await fetch(`/api/discovery/cancel/${encodeURIComponent(jobId)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.message || `خطا در لغو فرآیند کشف (${response.status})`);
        }
        return await response.json();
    },

    /**
     * Start Extraction job
     * @param {Object} config 
     * @returns {Promise<Object>}
     */
    async startExtraction(config) {
        const response = await fetch('/api/extraction/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.message || `خطا در شروع استخراج محتوا (${response.status})`);
        }
        const data = await response.json();
        if (data && data.data && data.data.job_id && !data.job_id) {
            data.job_id = data.data.job_id;
        }
        return data;
    },

    /**
     * Get Extraction job status
     * @param {string} jobId 
     * @returns {Promise<Object>}
     */
    async getExtractionStatus(jobId) {
        const response = await fetch(`/api/extraction/status/${encodeURIComponent(jobId)}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.message || `خطا در دریافت وضعیت استخراج (${response.status})`);
        }
        return await response.json();
    },

    /**
     * Cancel Extraction job
     * @param {string} jobId 
     * @returns {Promise<Object>}
     */
    async cancelExtraction(jobId) {
        const response = await fetch(`/api/extraction/cancel/${encodeURIComponent(jobId)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.message || `خطا در لغو عملیات استخراج (${response.status})`);
        }
        return await response.json();
    },

    /**
     * Retry Failed items in Extraction job
     * @param {string} jobId 
     * @returns {Promise<Object>}
     */
    async retryFailed(jobId) {
        const response = await fetch(`/api/extraction/retry/${encodeURIComponent(jobId)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.message || `خطا در تلاش مجدد موارد ناموفق (${response.status})`);
        }
        const data = await response.json();
        if (data && data.data && data.data.job_id && !data.job_id) {
            data.job_id = data.data.job_id;
        }
        return data;
    },

    /**
     * Export extracted documents
     * @param {string} jobId 
     * @param {string} format ('json' | 'jsonl' | 'markdown' | 'text')
     * @returns {Promise<Blob>}
     */
    async exportDocuments(jobId, format) {
        const response = await fetch('/api/export/download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                job_id: jobId,
                format: format
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || errData.message || `خطا در دریافت فایل خروجی (${response.status})`);
        }
        return await response.blob();
    }
};

// ============================================================================
// 3. Tab Management
// ============================================================================
/**
 * Switch active visible tab between 'discover' and 'extract'
 * @param {string} tabName 
 */
function switchTab(tabName) {
    if (state.activeTab === tabName) return;

    state.activeTab = tabName;

    // Toggle Tab Buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        const isTarget = btn.getAttribute('data-tab') === tabName;
        btn.classList.toggle('active', isTarget);
        btn.setAttribute('aria-selected', isTarget ? 'true' : 'false');
    });

    // Toggle Tab Content
    document.querySelectorAll('.tab-content').forEach(content => {
        const isTarget = content.id === `tab-content-${tabName}`;
        content.classList.toggle('active', isTarget);
    });
}

// ============================================================================
// 4. Discovery Functions
// ============================================================================
/**
 * Start the website discovery crawl
 */
async function startDiscovery() {
    if (state.isDiscovering) return;

    const rootUrlInput = document.getElementById('input-root-url');
    const rootUrl = rootUrlInput ? rootUrlInput.value.trim() : '';

    if (!rootUrl) {
        showToast('لطفاً آدرس ریشه وب‌سایت را وارد نمایید.', 'warning');
        if (rootUrlInput) rootUrlInput.focus();
        return;
    }

    // Basic URL validation
    try {
        new URL(rootUrl);
    } catch (_) {
        showToast('نشانی اینترنتی وارد شده معتبر نمی‌باشد (باید با http:// یا https:// آغاز شود).', 'error');
        if (rootUrlInput) rootUrlInput.focus();
        return;
    }

    // Collect settings
    const maxDepth = parseInt(document.getElementById('input-max-depth')?.value || '3', 10);
    const maxPages = parseInt(document.getElementById('input-max-pages')?.value || '100', 10);
    const timeout = parseInt(document.getElementById('input-timeout')?.value || '30', 10);
    const concurrency = parseInt(document.getElementById('input-concurrency')?.value || '5', 10);
    const delay = parseFloat(document.getElementById('input-delay')?.value || '0.5');
    const allowSubdomains = document.getElementById('check-allow-subdomains')?.checked || false;
    const keepQueryParams = document.getElementById('check-keep-query-params')?.checked || false;
    const useSitemap = document.getElementById('check-use-sitemap')?.checked || false;
    const pathPrefix = document.getElementById('input-path-prefix')?.value.trim() || '';

    const includePatternsRaw = document.getElementById('input-include-patterns')?.value || '';
    const excludePatternsRaw = document.getElementById('input-exclude-patterns')?.value || '';

    const includePatterns = includePatternsRaw.split('\n').map(s => s.trim()).filter(Boolean);
    const excludePatterns = excludePatternsRaw.split('\n').map(s => s.trim()).filter(Boolean);

    const config = {
        root_url: rootUrl,
        max_depth: maxDepth,
        max_pages: maxPages,
        timeout: timeout,
        concurrency: concurrency,
        delay: delay,
        allow_subdomains: allowSubdomains,
        keep_query_params: keepQueryParams,
        use_sitemap: useSitemap,
        include_patterns: includePatterns,
        exclude_patterns: excludePatterns,
        path_prefix: pathPrefix
    };

    // Update UI for starting
    state.isDiscovering = true;
    toggleDiscoveryControls(true);
    setDiscoveryStatusBadge('running', 'در حال پردازش...');
    showToast('فرآیند کشف صفحات آغاز شد.', 'info');

    try {
        const response = await api.startDiscovery(config);
        if (response && response.success === false) {
            throw new Error(response.message || 'خطا در آغاز کشف صفحات');
        }
        const jobId = response.data?.job_id || response.job_id || response.id;

        if (!jobId) {
            throw new Error('شناسه کاری از سمت سرور بازگردانده نشد.');
        }

        state.discoveryJobId = jobId;
        const jobIdEl = document.getElementById('discovery-job-id');
        if (jobIdEl) jobIdEl.textContent = jobId;

        // Reset previous discoveries
        state.discoveredPages = [];
        state.selectedUrls.clear();
        updateSelectedCounter();
        renderDiscoveryResults([]);

        // Start polling for progress
        startPolling(jobId, 'discovery', 2000);

    } catch (error) {
        state.isDiscovering = false;
        toggleDiscoveryControls(false);
        setDiscoveryStatusBadge('failed', 'خطا در شروع');
        showToast(error.message, 'error');
        showErrorModal('خطا در آغاز خزش', error.message);
    }
}

/**
 * Cancel the active discovery job
 */
async function cancelDiscovery() {
    if (!state.discoveryJobId || !state.isDiscovering) return;

    try {
        showToast('درخواست توقف خزش ارسال شد...', 'warning');
        await api.cancelDiscovery(state.discoveryJobId);
        stopPolling();
        state.isDiscovering = false;
        toggleDiscoveryControls(false);
        setDiscoveryStatusBadge('cancelled', 'متوقف شد');
        showToast('فرآیند کشف صفحات توسط کاربر متوقف شد.', 'info');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

/**
 * Update discovery progress bar, stats, and results table
 * @param {Object} statusData 
 */
function updateDiscoveryProgress(statusData) {
    if (!statusData) return;

    const totalDiscovered = statusData.total_discovered ?? statusData.discovered ?? statusData.total ?? (Array.isArray(statusData.pages) ? statusData.pages.length : 0);
    const crawled = statusData.crawled ?? statusData.total_crawled ?? statusData.completed ?? 0;
    const failed = statusData.failed ?? statusData.total_failed ?? 0;
    const jobStatus = (statusData.status || 'running').toLowerCase();

    // Update stat numbers
    const totalEl = document.getElementById('stat-discovery-total');
    const crawledEl = document.getElementById('stat-discovery-crawled');
    const failedEl = document.getElementById('stat-discovery-failed');

    if (totalEl) totalEl.textContent = totalDiscovered.toLocaleString('fa-IR');
    if (crawledEl) crawledEl.textContent = crawled.toLocaleString('fa-IR');
    if (failedEl) failedEl.textContent = failed.toLocaleString('fa-IR');

    // Update badge count on tab header
    const tabCountBadge = document.getElementById('discover-badge-count');
    if (tabCountBadge) tabCountBadge.textContent = totalDiscovered;

    // Calculate percentage
    let percent = 0;
    if (typeof statusData.progress === 'number') {
        percent = Math.min(100, Math.max(0, Math.round(statusData.progress)));
    } else if (totalDiscovered > 0) {
        percent = Math.min(100, Math.round(((crawled + failed) / totalDiscovered) * 100));
    }

    const progressBar = document.getElementById('discovery-progress-bar');
    const progressText = document.getElementById('discovery-progress-text');
    if (progressBar) progressBar.style.width = `${percent}%`;
    if (progressText) progressText.textContent = `${percent}%`;

    // Update pages table
    const pages = statusData.pages || statusData.results || [];
    if (Array.isArray(pages) && pages.length > 0) {
        state.discoveredPages = pages;
        renderDiscoveryResults(filterPages(pages, state.searchQuery));
    }

    // Check completion status
    if (['completed', 'finished', 'done', 'success'].includes(jobStatus)) {
        stopPolling();
        state.isDiscovering = false;
        toggleDiscoveryControls(false);
        setDiscoveryStatusBadge('success', 'پایان موفقیت‌آمیز');
        if (progressBar) progressBar.style.width = '100%';
        if (progressText) progressText.textContent = '100%';
        showToast(`فرآیند کشف به پایان رسید. تعداد ${totalDiscovered} صفحه کشف شد.`, 'success');
    } else if (['failed', 'error'].includes(jobStatus)) {
        stopPolling();
        state.isDiscovering = false;
        toggleDiscoveryControls(false);
        setDiscoveryStatusBadge('failed', 'خطا در فرآیند');
        showToast(statusData.error || 'خطایی در اجرای فرآیند کشف رخ داده است.', 'error');
    } else if (['cancelled', 'stopped'].includes(jobStatus)) {
        stopPolling();
        state.isDiscovering = false;
        toggleDiscoveryControls(false);
        setDiscoveryStatusBadge('cancelled', 'متوقف شد');
    } else {
        setDiscoveryStatusBadge('running', 'در حال خزش...');
    }
}

/**
 * Render Discovery results table
 * @param {Array<Object>} pages 
 */
function renderDiscoveryResults(pages) {
    const tbody = document.getElementById('discovery-table-body');
    const showingCountEl = document.getElementById('discovery-showing-count');
    const totalCountEl = document.getElementById('discovery-total-count');

    if (totalCountEl) totalCountEl.textContent = state.discoveredPages.length.toLocaleString('fa-IR');
    if (showingCountEl) showingCountEl.textContent = pages.length.toLocaleString('fa-IR');

    if (!tbody) return;

    if (!pages || pages.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-state-row">
                <td colspan="6">
                    <div class="empty-state">
                        <div class="empty-icon"><i class="fa-solid fa-folder-open"></i></div>
                        <h4>${state.searchQuery ? 'هیچ نتیجه‌ای مطابق جستجو یافت نشد' : 'هیچ صفحه‌ای هنوز کشف نشده است'}</h4>
                        <p>${state.searchQuery ? 'عبارت جستجو را تغییر دهید یا فیلتر را پاک نمایید.' : 'آدرس ریشه را وارد کرده و فرآیند کشف را آغاز نمایید.'}</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    let rowsHtml = '';
    pages.forEach(page => {
        const url = page.url || '';
        const title = page.title || '(بدون عنوان)';
        const pageStatus = (page.status || 'pending').toLowerCase();
        const source = page.source || 'crawl';
        const depth = page.depth !== undefined ? page.depth : '-';
        const isChecked = state.selectedUrls.has(url);

        let badgeClass = 'badge-secondary';
        let statusLabel = pageStatus;

        if (['success', 'completed', 'crawled', '200', 'ok'].includes(pageStatus)) {
            badgeClass = 'badge-success';
            statusLabel = 'پیمایش شده';
        } else if (pageStatus === 'redirect_loop') {
            badgeClass = 'badge-warning';
            statusLabel = 'حلقه ریدایرکت';
        } else if (pageStatus === 'too_many_redirects') {
            badgeClass = 'badge-warning';
            statusLabel = 'سقف ریدایرکت';
        } else if (['failed', 'error', '404', '500'].includes(pageStatus)) {
            badgeClass = 'badge-failed';
            statusLabel = 'خطا';
        } else if (['blocked', '403', 'robot'].includes(pageStatus)) {
            badgeClass = 'badge-blocked';
            statusLabel = 'مسدود';
        } else if (['pending', 'queued', 'in_progress', '0'].includes(pageStatus)) {
            badgeClass = 'badge-pending';
            statusLabel = 'در انتظار';
        }

        rowsHtml += `
            <tr data-url="${escapeHtml(url)}">
                <td class="col-checkbox">
                    <input type="checkbox" class="row-checkbox discovery-row-checkbox" data-url="${escapeHtml(url)}" ${isChecked ? 'checked' : ''} />
                </td>
                <td class="col-url" title="${escapeHtml(url)}">
                    <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a>
                </td>
                <td class="col-title" title="${escapeHtml(title)}">${escapeHtml(title)}</td>
                <td class="col-status">
                    <span class="badge ${badgeClass}">${escapeHtml(statusLabel)}</span>
                </td>
                <td class="col-source">
                    <span class="source-tag">${escapeHtml(source)}</span>
                </td>
                <td class="col-depth">${depth}</td>
            </tr>
        `;
    });

    tbody.innerHTML = rowsHtml;

    // Attach row checkbox event listeners
    tbody.querySelectorAll('.discovery-row-checkbox').forEach(cb => {
        cb.addEventListener('change', (e) => {
            const rowUrl = e.target.getAttribute('data-url');
            if (rowUrl) toggleSelection(rowUrl, e.target.checked);
        });
    });

    updateHeaderCheckbox();
}

/**
 * Filter pages array based on query text
 * @param {Array<Object>} pages 
 * @param {string} query 
 * @returns {Array<Object>}
 */
function filterPages(pages, query) {
    if (!query) return pages;
    const lowerQuery = query.toLowerCase();
    return pages.filter(p => {
        const u = (p.url || '').toLowerCase();
        const t = (p.title || '').toLowerCase();
        return u.includes(lowerQuery) || t.includes(lowerQuery);
    });
}

/**
 * Toggle enable/disable discovery controls
 * @param {boolean} isRunning 
 */
function toggleDiscoveryControls(isRunning) {
    const btnStart = document.getElementById('btn-start-discovery');
    const btnCancel = document.getElementById('btn-cancel-discovery');
    const rootUrlInput = document.getElementById('input-root-url');

    if (btnStart) {
        btnStart.disabled = isRunning;
        btnStart.classList.toggle('hidden', isRunning);
    }
    if (btnCancel) {
        btnCancel.classList.toggle('hidden', !isRunning);
    }
    if (rootUrlInput) {
        rootUrlInput.disabled = isRunning;
    }
}

/**
 * Set status badge text and style for discovery
 * @param {string} statusType ('running' | 'success' | 'failed' | 'cancelled')
 * @param {string} label 
 */
function setDiscoveryStatusBadge(statusType, label) {
    const badge = document.getElementById('discovery-status-badge');
    if (!badge) return;

    badge.className = 'badge';
    if (statusType === 'running') badge.classList.add('badge-pending');
    else if (statusType === 'success') badge.classList.add('badge-success');
    else if (statusType === 'failed') badge.classList.add('badge-failed');
    else if (statusType === 'cancelled') badge.classList.add('badge-blocked');
    else badge.classList.add('badge-secondary');

    badge.textContent = label;
}

// ============================================================================
// 5. Selection Functions
// ============================================================================
/**
 * Select all currently filtered pages
 */
function selectAll() {
    const filtered = filterPages(state.discoveredPages, state.searchQuery);
    filtered.forEach(page => {
        if (page.url) state.selectedUrls.add(page.url);
    });

    document.querySelectorAll('.discovery-row-checkbox').forEach(cb => {
        cb.checked = true;
    });

    updateSelectedCounter();
    updateHeaderCheckbox();
    showToast(`تعداد ${filtered.length} صفحه انتخاب گردید.`, 'info');
}

/**
 * Deselect all pages
 */
function deselectAll() {
    state.selectedUrls.clear();

    document.querySelectorAll('.discovery-row-checkbox').forEach(cb => {
        cb.checked = false;
    });

    updateSelectedCounter();
    updateHeaderCheckbox();
    showToast('انتخاب تمام صفحات لغو شد.', 'info');
}

/**
 * Toggle individual URL selection
 * @param {string} url 
 * @param {boolean} forceCheck 
 */
function toggleSelection(url, forceCheck) {
    if (!url) return;
    if (forceCheck !== undefined) {
        if (forceCheck) state.selectedUrls.add(url);
        else state.selectedUrls.delete(url);
    } else {
        if (state.selectedUrls.has(url)) state.selectedUrls.delete(url);
        else state.selectedUrls.add(url);
    }

    updateSelectedCounter();
    updateHeaderCheckbox();
}

/**
 * Update selected count badges across UI
 */
function updateSelectedCounter() {
    const count = state.selectedUrls.size;
    const statSelectedEl = document.getElementById('stat-discovery-selected');
    const sendCountEl = document.getElementById('btn-send-count');
    const btnSend = document.getElementById('btn-send-to-extraction');

    if (statSelectedEl) statSelectedEl.textContent = count.toLocaleString('fa-IR');
    if (sendCountEl) sendCountEl.textContent = count;
    if (btnSend) btnSend.disabled = (count === 0);
}

/**
 * Update header checkbox state based on visible row selections
 */
function updateHeaderCheckbox() {
    const headerCb = document.getElementById('header-discovery-checkbox');
    if (!headerCb) return;

    const visibleCheckboxes = document.querySelectorAll('.discovery-row-checkbox');
    if (visibleCheckboxes.length === 0) {
        headerCb.checked = false;
        headerCb.indeterminate = false;
        return;
    }

    let checkedCount = 0;
    visibleCheckboxes.forEach(cb => {
        if (cb.checked) checkedCount++;
    });

    if (checkedCount === 0) {
        headerCb.checked = false;
        headerCb.indeterminate = false;
    } else if (checkedCount === visibleCheckboxes.length) {
        headerCb.checked = true;
        headerCb.indeterminate = false;
    } else {
        headerCb.checked = false;
        headerCb.indeterminate = true;
    }
}

/**
 * Transfer selected URLs to the Extraction tab and switch view
 */
function sendToExtraction() {
    if (state.selectedUrls.size === 0) {
        showToast('هیچ نشانی‌ای برای ارسال انتخاب نشده است.', 'warning');
        return;
    }

    // Add selected URLs to the extraction set
    state.selectedUrls.forEach(url => {
        state.extractionUrls.add(url);
    });

    renderExtractionUrlList();
    showToast(`${state.selectedUrls.size} نشانی با موفقیت به بخش استخراج اضافه گردید.`, 'success');

    // Switch to Extraction tab
    switchTab('extract');
}

// ============================================================================
// 6. Extraction Functions
// ============================================================================
/**
 * Add URLs typed or pasted into the manual textarea
 */
function addManualExtractionUrls() {
    const textarea = document.getElementById('input-extraction-urls-textarea');
    if (!textarea) return;

    const parsedUrls = parseUrlsFromTextarea(textarea.value);
    if (parsedUrls.length === 0) {
        showToast('لطفاً حداقل یک نشانی معتبر اینترنتی وارد فرمایید.', 'warning');
        return;
    }

    let addedCount = 0;
    parsedUrls.forEach(url => {
        if (!state.extractionUrls.has(url)) {
            state.extractionUrls.add(url);
            addedCount++;
        }
    });

    textarea.value = '';
    const badge = document.getElementById('urls-input-count');
    if (badge) badge.textContent = '0 مورد';

    renderExtractionUrlList();
    showToast(`${addedCount} نشانی جدید به فهرست استخراج اضافه شد.`, 'success');
}

/**
 * Remove an individual URL from the extraction queue
 * @param {string} url 
 */
function removeExtractionUrl(url) {
    if (state.extractionUrls.has(url)) {
        state.extractionUrls.delete(url);
        renderExtractionUrlList();
    }
}

/**
 * Clear all selected URLs in the extraction queue
 */
function clearAllExtractionUrls() {
    if (state.extractionUrls.size === 0) return;
    state.extractionUrls.clear();
    renderExtractionUrlList();
    showToast('فهرست نشانی‌های استخراج به طور کامل پاک شد.', 'info');
}

/**
 * Render the list of URLs queued for extraction
 */
function renderExtractionUrlList() {
    const container = document.getElementById('selected-urls-list-container');
    const countBadge = document.getElementById('selected-urls-count');
    const tabExtractBadge = document.getElementById('extract-badge-count');

    const total = state.extractionUrls.size;
    if (countBadge) countBadge.textContent = `${total} نشانی`;
    if (tabExtractBadge) tabExtractBadge.textContent = total;

    if (!container) return;

    if (total === 0) {
        container.innerHTML = `<div class="empty-list-notice">هیچ نشانی‌ای هنوز جهت استخراج اضافه نشده است. می‌توانید از بخش کشف صفحات مواردی را ارسال کنید یا به‌صورت دستی نشانی اضافه نمایید.</div>`;
        return;
    }

    let html = '';
    state.extractionUrls.forEach(url => {
        html += `
            <div class="url-item-chip" title="${escapeHtml(url)}">
                <span class="url-item-text">${escapeHtml(url)}</span>
                <button type="button" class="btn-remove-url" data-url="${escapeHtml(url)}" title="حذف این نشانی">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>
        `;
    });

    container.innerHTML = html;

    // Attach remove handlers
    container.querySelectorAll('.btn-remove-url').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetUrl = btn.getAttribute('data-url');
            if (targetUrl) removeExtractionUrl(targetUrl);
        });
    });
}

/**
 * Start the content extraction process
 */
async function startExtraction() {
    if (state.isExtracting) return;

    // If textarea has unadded URLs, try to add them automatically
    const textarea = document.getElementById('input-extraction-urls-textarea');
    if (textarea && textarea.value.trim()) {
        const extraUrls = parseUrlsFromTextarea(textarea.value);
        extraUrls.forEach(u => state.extractionUrls.add(u));
        textarea.value = '';
        renderExtractionUrlList();
    }

    if (state.extractionUrls.size === 0) {
        showToast('لطفاً ابتدا حداقل یک نشانی جهت استخراج اضافه نمایید.', 'warning');
        return;
    }

    const category = document.getElementById('input-extraction-category')?.value.trim() || '';
    const strategy = document.getElementById('select-extraction-strategy')?.value || 'auto';
    const crawlMode = document.getElementById('select-crawl-mode')?.value || 'single_page';
    const renderMode = document.getElementById('select-render-mode')?.value || 'auto';
    const minQuality = parseInt(document.getElementById('input-min-quality-score')?.value || '40', 10);
    const concurrency = parseInt(document.getElementById('input-extraction-concurrency')?.value || '5', 10);

    const config = {
        urls: Array.from(state.extractionUrls),
        category: category,
        extraction_strategy: strategy,
        extractor: strategy,
        strategy: strategy,
        crawl_mode: crawlMode,
        render_mode: renderMode,
        min_quality_score: minQuality,
        max_concurrency: concurrency,
        concurrency: concurrency
    };

    state.isExtracting = true;
    toggleExtractionControls(true);
    setExtractionStatusBadge('running', 'در حال استخراج...');
    showToast('عملیات استخراج محتوا آغاز گردید.', 'info');

    try {
        const response = await api.startExtraction(config);
        if (response && response.success === false) {
            throw new Error(response.message || 'خطا در آغاز استخراج محتوا');
        }
        const jobId = response.data?.job_id || response.job_id || response.id;

        if (!jobId) {
            throw new Error('شناسه کاری از سمت سرور بازگردانده نشد.');
        }

        state.extractionJobId = jobId;
        const jobIdEl = document.getElementById('extraction-job-id');
        if (jobIdEl) jobIdEl.textContent = jobId;

        // Reset previous extracted documents
        state.extractedDocuments = [];
        renderExtractionResults([]);

        // Disable export buttons until completion
        toggleExportButtons(false);

        // Start polling
        startPolling(jobId, 'extraction', 2000);

    } catch (error) {
        state.isExtracting = false;
        toggleExtractionControls(false);
        setExtractionStatusBadge('failed', 'خطا در شروع');
        showToast(error.message, 'error');
        showErrorModal('خطا در آغاز استخراج', error.message);
    }
}

/**
 * Cancel the active extraction job
 */
async function cancelExtraction() {
    if (!state.extractionJobId || !state.isExtracting) return;

    try {
        showToast('درخواست توقف استخراج ارسال شد...', 'warning');
        await api.cancelExtraction(state.extractionJobId);
        stopPolling();
        state.isExtracting = false;
        toggleExtractionControls(false);
        setExtractionStatusBadge('cancelled', 'متوقف شد');
        showToast('فرآیند استخراج محتوا لغو گردید.', 'info');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

/**
 * Retry failed documents in current extraction job
 */
async function retryFailedExtraction() {
    if (!state.extractionJobId) return;

    try {
        showToast('درخواست تلاش مجدد موارد ناموفق ارسال شد...', 'info');
        const res = await api.retryFailed(state.extractionJobId);
        if (res && res.success === false) {
            throw new Error(res.message || 'خطا در تلاش مجدد موارد ناموفق');
        }
        const newJobId = res.data?.job_id || res.job_id || state.extractionJobId;
        state.extractionJobId = newJobId;
        const jobIdEl = document.getElementById('extraction-job-id');
        if (jobIdEl) jobIdEl.textContent = newJobId;

        state.isExtracting = true;
        toggleExtractionControls(true);
        setExtractionStatusBadge('running', 'تلاش مجدد موارد ناموفق...');
        startPolling(newJobId, 'extraction', 2000);
    } catch (error) {
        showToast(error.message, 'error');
    }
}

/**
 * Update Extraction progress bar, stats, and results table
 * @param {Object} statusData 
 */
function updateExtractionProgress(statusData) {
    if (!statusData) return;

    const total = statusData.total ?? state.extractionUrls.size ?? 0;
    const completed = statusData.completed ?? 0;
    const failed = statusData.failed ?? 0;
    const jobStatus = (statusData.status || 'running').toLowerCase();

    // Update stats
    const totalEl = document.getElementById('stat-extraction-total');
    const compEl = document.getElementById('stat-extraction-completed');
    const failEl = document.getElementById('stat-extraction-failed');
    const wordsEl = document.getElementById('stat-extraction-words');

    if (totalEl) totalEl.textContent = total.toLocaleString('fa-IR');
    if (compEl) compEl.textContent = completed.toLocaleString('fa-IR');
    if (failEl) failEl.textContent = failed.toLocaleString('fa-IR');

    // Calculate total words
    const docs = statusData.documents || statusData.results || [];
    let totalWords = 0;
    if (Array.isArray(docs)) {
        totalWords = docs.reduce((acc, doc) => acc + (doc.word_count || 0), 0);
    }
    if (wordsEl) wordsEl.textContent = totalWords.toLocaleString('fa-IR');

    // Progress bar calculation
    let percent = 0;
    if (typeof statusData.progress === 'number') {
        percent = Math.min(100, Math.max(0, Math.round(statusData.progress)));
    } else if (total > 0) {
        percent = Math.min(100, Math.round(((completed + failed) / total) * 100));
    }

    const progressBar = document.getElementById('extraction-progress-bar');
    const progressText = document.getElementById('extraction-progress-text');
    if (progressBar) progressBar.style.width = `${percent}%`;
    if (progressText) progressText.textContent = `${percent}%`;

    // Render results
    if (Array.isArray(docs) && docs.length > 0) {
        state.extractedDocuments = docs;
        renderExtractionResults(docs);
    }

    // Toggle Retry button if failed items exist
    const btnRetry = document.getElementById('btn-retry-failed');
    if (btnRetry) {
        btnRetry.classList.toggle('hidden', failed === 0 || state.isExtracting);
    }

    // Check completion status
    if (['completed', 'finished', 'done', 'success'].includes(jobStatus)) {
        stopPolling();
        state.isExtracting = false;
        toggleExtractionControls(false);
        setExtractionStatusBadge('success', 'استخراج پایان یافت');
        toggleExportButtons(true);
        if (progressBar) progressBar.style.width = '100%';
        if (progressText) progressText.textContent = '100%';
        showToast(`استخراج با موفقیت پایان یافت. (${completed} موفق، ${failed} ناموفق)`, 'success');
    } else if (['failed', 'error'].includes(jobStatus)) {
        stopPolling();
        state.isExtracting = false;
        toggleExtractionControls(false);
        setExtractionStatusBadge('failed', 'خطا در استخراج');
        toggleExportButtons(docs.length > 0);
        showToast(statusData.error || 'خطایی در اجرای فرآیند استخراج رخ داد.', 'error');
    } else if (['cancelled', 'stopped'].includes(jobStatus)) {
        stopPolling();
        state.isExtracting = false;
        toggleExtractionControls(false);
        setExtractionStatusBadge('cancelled', 'متوقف شد');
        toggleExportButtons(docs.length > 0);
    } else {
        setExtractionStatusBadge('running', 'در حال پردازش...');
    }
}

/**
 * Render Extraction results table
 * @param {Array<Object>} docs 
 */
function renderExtractionResults(docs) {
    const tbody = document.getElementById('extraction-table-body');
    const showingCountEl = document.getElementById('extraction-showing-count');

    if (showingCountEl) showingCountEl.textContent = (docs || []).length.toLocaleString('fa-IR');
    if (!tbody) return;

    if (!docs || docs.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-state-row">
                <td colspan="6">
                    <div class="empty-state">
                        <div class="empty-icon"><i class="fa-solid fa-newspaper"></i></div>
                        <h4>هیچ صفحه‌ای هنوز استخراج نشده است</h4>
                        <p>نشانی‌های مورد نظرتان را اضافه کرده و روی «شروع عملیات استخراج» کلیک فرمایید.</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    let rowsHtml = '';
    docs.forEach((doc, idx) => {
        const url = doc.url || '';
        const title = doc.title || '(بدون عنوان)';
        const docStatus = (doc.status || 'pending').toLowerCase();
        const wordCount = doc.word_count !== undefined ? doc.word_count.toLocaleString('fa-IR') : '0';

        // Engine and Page Type Badges
        const engine = (doc.extractor_used || doc.extraction_method || 'auto').toLowerCase();
        let engineBadge = `<span class="badge badge-secondary" style="font-size: 11px;"><i class="fa-solid fa-gear"></i> ${escapeHtml(engine)}</span>`;
        if (engine.includes('trafilatura')) {
            engineBadge = `<span class="badge badge-info" style="font-size: 11px;"><i class="fa-solid fa-feather"></i> Trafilatura</span>`;
        } else if (engine.includes('beautifulsoup') || engine.includes('bs4')) {
            engineBadge = `<span class="badge badge-warning" style="font-size: 11px;"><i class="fa-solid fa-code"></i> BS4</span>`;
        } else if (engine.includes('crawl4ai')) {
            engineBadge = `<span class="badge badge-primary" style="font-size: 11px;"><i class="fa-solid fa-robot"></i> Crawl4AI</span>`;
        }

        const pageType = (doc.page_type || '').toLowerCase();
        const typeBadge = pageType && pageType !== 'unknown' 
            ? `<span class="badge badge-outline" style="font-size: 10px; margin-right: 4px;">${escapeHtml(pageType)}</span>` 
            : '';

        // Quality formatting from quality_report.score or doc.quality
        let qScore = null;
        if (doc.quality_report && typeof doc.quality_report.score === 'number') {
            qScore = doc.quality_report.score;
        } else if (typeof doc.quality === 'number') {
            qScore = Math.round(doc.quality <= 1.0 ? doc.quality * 100 : doc.quality);
        }

        let qualityTag = '<span class="quality-meter quality-medium">-</span>';
        if (qScore !== null) {
            if (qScore >= 70) qualityTag = `<span class="quality-meter quality-high">${qScore}/100</span>`;
            else if (qScore >= 40) qualityTag = `<span class="quality-meter quality-medium">${qScore}/100</span>`;
            else qualityTag = `<span class="quality-meter quality-low">${qScore}/100</span>`;
        }

        let badgeClass = 'badge-secondary';
        let statusLabel = docStatus;
        if (['success', 'completed', 'ok'].includes(docStatus)) {
            badgeClass = 'badge-success';
            statusLabel = 'استخراج شد';
        } else if (['failed', 'error'].includes(docStatus)) {
            badgeClass = 'badge-failed';
            statusLabel = 'خطا';
        } else {
            badgeClass = 'badge-pending';
            statusLabel = 'در حال انجام';
        }

        rowsHtml += `
            <tr>
                <td class="col-url" title="${escapeHtml(url)}">
                    <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a>
                </td>
                <td class="col-title" title="${escapeHtml(title)}">${escapeHtml(title)}</td>
                <td class="col-status">
                    <span class="badge ${badgeClass}">${escapeHtml(statusLabel)}</span>
                </td>
                <td class="col-engine">
                    ${engineBadge} ${typeBadge}
                </td>
                <td class="col-words">${wordCount}</td>
                <td class="col-quality">${qualityTag}</td>
                <td class="col-actions">
                    <button type="button" class="btn btn-outline btn-sm btn-preview-doc" data-idx="${idx}" title="مشاهده محتوای استخراج شده">
                        <i class="fa-solid fa-eye"></i>
                        <span>مشاهده</span>
                    </button>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = rowsHtml;

    // Attach click event for preview
    tbody.querySelectorAll('.btn-preview-doc').forEach(btn => {
        btn.addEventListener('click', () => {
            const docIndex = parseInt(btn.getAttribute('data-idx') || '0', 10);
            const doc = docs[docIndex];
            if (doc) openPreviewModal(doc);
        });
    });
}

/**
 * Toggle enable/disable extraction controls
 * @param {boolean} isRunning 
 */
function toggleExtractionControls(isRunning) {
    const btnStart = document.getElementById('btn-start-extraction');
    const btnCancel = document.getElementById('btn-cancel-extraction');
    const btnAddUrls = document.getElementById('btn-add-urls');
    const btnClearUrls = document.getElementById('btn-clear-selected-urls');

    if (btnStart) {
        btnStart.disabled = isRunning;
        btnStart.classList.toggle('hidden', isRunning);
    }
    if (btnCancel) {
        btnCancel.classList.toggle('hidden', !isRunning);
    }
    if (btnAddUrls) btnAddUrls.disabled = isRunning;
    if (btnClearUrls) btnClearUrls.disabled = isRunning;
}

/**
 * Set status badge text and style for extraction
 * @param {string} statusType 
 * @param {string} label 
 */
function setExtractionStatusBadge(statusType, label) {
    const badge = document.getElementById('extraction-status-badge');
    if (!badge) return;

    badge.className = 'badge';
    if (statusType === 'running') badge.classList.add('badge-pending');
    else if (statusType === 'success') badge.classList.add('badge-success');
    else if (statusType === 'failed') badge.classList.add('badge-failed');
    else if (statusType === 'cancelled') badge.classList.add('badge-blocked');
    else badge.classList.add('badge-secondary');

    badge.textContent = label;
}

/**
 * Toggle export buttons disabled state
 * @param {boolean} enabled 
 */
function toggleExportButtons(enabled) {
    document.querySelectorAll('.btn-export').forEach(btn => {
        btn.disabled = !enabled;
    });
}

// ============================================================================
// 7. Export Functions
// ============================================================================
/**
 * Export extracted documents in specified format (json, jsonl, markdown, text)
 * @param {string} format 
 */
async function exportResults(format) {
    if (!state.extractionJobId) {
        showToast('شناسه کار استخراج یافت نشد.', 'warning');
        return;
    }

    try {
        showToast(`در حال تولید و آماده‌سازی فایل خروجی ${format.toUpperCase()}...`, 'info');
        
        let blob;
        try {
            // Attempt server-side export API
            blob = await api.exportDocuments(state.extractionJobId, format);
        } catch (apiError) {
            // Fallback to client-side generation if server export endpoint is unavailable
            console.warn('Server export endpoint failed, generating file locally...', apiError);
            blob = generateLocalExportBlob(state.extractedDocuments, format);
        }

        const extMap = {
            json: 'json',
            jsonl: 'jsonl',
            markdown: 'md',
            text: 'txt'
        };
        const fileExt = extMap[format] || format;
        const filename = `crawlbot_export_${state.extractionJobId.slice(0, 8)}_${new Date().toISOString().slice(0, 10)}.${fileExt}`;

        triggerBlobDownload(blob, filename);
        showToast(`فایل خروجی با پسوند ${fileExt.toUpperCase()} با موفقیت دانلود شد.`, 'success');

    } catch (error) {
        showToast(`خطا در دریافت خروجی: ${error.message}`, 'error');
    }
}

/**
 * Export discovered URLs from Discovery tab
 */
function exportDiscoveredUrls() {
    if (state.discoveredPages.length === 0) {
        showToast('هیچ صفحه‌ای برای خروجی گرفتن وجود ندارد.', 'warning');
        return;
    }

    // If items are selected, export selected; otherwise export all
    const urlsToExport = state.selectedUrls.size > 0
        ? Array.from(state.selectedUrls)
        : state.discoveredPages.map(p => p.url).filter(Boolean);

    const content = urlsToExport.join('\n');
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const filename = `discovered_urls_${new Date().toISOString().slice(0, 10)}.txt`;

    triggerBlobDownload(blob, filename);
    showToast(`تعداد ${urlsToExport.length} نشانی ذخیره شد.`, 'success');
}

/**
 * Fallback client-side export generator
 * @param {Array<Object>} docs 
 * @param {string} format 
 * @returns {Blob}
 */
function generateLocalExportBlob(docs, format) {
    let content = '';
    let mimeType = 'text/plain;charset=utf-8';

    if (format === 'json') {
        content = JSON.stringify(docs, null, 2);
        mimeType = 'application/json;charset=utf-8';
    } else if (format === 'jsonl') {
        content = docs.map(d => JSON.stringify(d)).join('\n');
        mimeType = 'application/x-ndjson;charset=utf-8';
    } else if (format === 'markdown') {
        content = docs.map(d => {
            return `# ${d.title || 'Untitled'}\n\n- **URL:** ${d.url}\n- **Date:** ${new Date().toISOString()}\n- **Words:** ${d.word_count || 0}\n\n---\n\n${d.content || d.text || ''}\n\n`;
        }).join('\n\n---\n\n');
        mimeType = 'text/markdown;charset=utf-8';
    } else { // text
        content = docs.map(d => {
            return `========================================\nURL: ${d.url}\nTITLE: ${d.title || ''}\n========================================\n\n${d.content || d.text || ''}\n\n`;
        }).join('\n\n');
        mimeType = 'text/plain;charset=utf-8';
    }

    return new Blob([content], { type: mimeType });
}

/**
 * Trigger file download via a created Blob URL
 * @param {Blob} blob 
 * @param {string} filename 
 */
function triggerBlobDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 200);
}

// ============================================================================
// 8. Utility Functions
// ============================================================================
/**
 * Show a toast notification
 * @param {string} message 
 * @param {'success' | 'error' | 'warning' | 'info'} type 
 * @param {number} duration 
 */
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let iconClass = 'fa-info-circle';
    if (type === 'success') iconClass = 'fa-circle-check';
    else if (type === 'error') iconClass = 'fa-triangle-exclamation';
    else if (type === 'warning') iconClass = 'fa-circle-exclamation';

    toast.innerHTML = `
        <i class="fa-solid ${iconClass} toast-icon"></i>
        <div class="toast-message">${escapeHtml(message)}</div>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-hiding');
        setTimeout(() => {
            if (toast.parentElement) toast.parentElement.removeChild(toast);
        }, 260);
    }, duration);
}

/**
 * Show or hide full-page/element loading indicator
 * @param {boolean} show 
 */
function showLoading(show) {
    const indicator = document.getElementById('system-status-indicator');
    const text = document.getElementById('system-status-text');
    if (!indicator || !text) return;

    if (show) {
        indicator.classList.remove('active');
        text.innerHTML = '<span class="spinner"></span> در حال انجام...';
    } else {
        indicator.classList.add('active');
        text.textContent = 'سیستم آماده است';
    }
}

/**
 * Truncate long URLs for display
 * @param {string} url 
 * @param {number} maxLength 
 * @returns {string}
 */
function formatUrl(url, maxLength = 50) {
    if (!url) return '';
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength - 3) + '...';
}

/**
 * Filter discovery results in real-time
 * @param {string} query 
 */
function filterResults(query) {
    state.searchQuery = (query || '').trim();
    const btnClear = document.getElementById('btn-clear-discovery-search');
    if (btnClear) {
        btnClear.classList.toggle('hidden', state.searchQuery === '');
    }

    const filtered = filterPages(state.discoveredPages, state.searchQuery);
    renderDiscoveryResults(filtered);
}

/**
 * Parse, clean, validate and deduplicate URLs from textarea
 * @param {string} text 
 * @returns {Array<string>}
 */
function parseUrlsFromTextarea(text) {
    if (!text) return [];

    const lines = text.split('\n');
    const validUrls = new Set();

    for (let line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        try {
            // Validate URL syntax
            const urlObj = new URL(trimmed);
            if (urlObj.protocol === 'http:' || urlObj.protocol === 'https:') {
                validUrls.add(trimmed);
            }
        } catch (_) {
            // Attempt prepend https:// if missing protocol
            if (trimmed.includes('.') && !trimmed.includes(' ')) {
                try {
                    const fallbackUrl = new URL('https://' + trimmed);
                    validUrls.add(fallbackUrl.href);
                } catch (__) {
                    // ignore invalid line
                }
            }
        }
    }

    return Array.from(validUrls);
}

/**
 * Open Preview Modal dialog for a document
 * @param {Object} doc 
 */
function openPreviewModal(doc) {
    const modal = document.getElementById('preview-modal');
    if (!modal) return;

    const urlEl = document.getElementById('modal-doc-url');
    const titleEl = document.getElementById('preview-modal-title');
    const statusEl = document.getElementById('modal-doc-status');
    const engineEl = document.getElementById('modal-doc-engine');
    const pageTypeEl = document.getElementById('modal-doc-page-type');
    const wordcountEl = document.getElementById('modal-doc-wordcount');
    const qualityEl = document.getElementById('modal-doc-quality');
    const contentEl = document.getElementById('modal-doc-content');

    if (urlEl) {
        urlEl.textContent = doc.url || '-';
        urlEl.href = doc.url || '#';
    }
    if (titleEl) titleEl.textContent = doc.title || 'پیش‌نمایش محتوای استخراج‌شده';
    if (statusEl) {
        statusEl.textContent = doc.status || 'موفق';
        statusEl.className = 'badge ' + (doc.status === 'failed' ? 'badge-failed' : 'badge-success');
    }
    if (engineEl) {
        engineEl.textContent = doc.extractor_used || doc.extraction_method || 'ناشناخته';
    }
    if (pageTypeEl) {
        pageTypeEl.textContent = doc.page_type || 'نامشخص';
    }
    if (wordcountEl) wordcountEl.textContent = (doc.word_count || 0).toLocaleString('fa-IR');

    let qDisplay = '-';
    if (doc.quality_report && typeof doc.quality_report.score === 'number') {
        qDisplay = `${doc.quality_report.score}/100`;
    } else if (doc.quality !== undefined) {
        qDisplay = typeof doc.quality === 'number' ? `${Math.round(doc.quality * 100)}%` : String(doc.quality);
    }
    if (qualityEl) qualityEl.textContent = qDisplay;

    if (contentEl) {
        const textToDisplay = doc.content_markdown || doc.content_text || doc.content || doc.text || '(محتوای متنی یافت نشد یا صفحه بدون متن اصلی است)';
        contentEl.textContent = textToDisplay;
    }

    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
}

/**
 * Close Preview Modal
 */
function closePreviewModal() {
    const modal = document.getElementById('preview-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.setAttribute('aria-hidden', 'true');
    }
}

/**
 * Show Error Details Modal
 * @param {string} title 
 * @param {string} message 
 * @param {string} [details] 
 */
function showErrorModal(title, message, details) {
    const modal = document.getElementById('error-modal');
    if (!modal) return;

    const titleEl = document.getElementById('error-modal-title');
    const msgEl = document.getElementById('error-modal-message');
    const detailsBox = document.getElementById('error-modal-details-box');
    const detailsCode = document.getElementById('error-modal-details');

    if (titleEl) titleEl.textContent = title;
    if (msgEl) msgEl.textContent = message;

    if (details && detailsBox && detailsCode) {
        detailsCode.textContent = typeof details === 'object' ? JSON.stringify(details, null, 2) : String(details);
        detailsBox.classList.remove('hidden');
    } else if (detailsBox) {
        detailsBox.classList.add('hidden');
    }

    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
}

/**
 * Close Error Modal
 */
function closeErrorModal() {
    const modal = document.getElementById('error-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.setAttribute('aria-hidden', 'true');
    }
}

/**
 * Basic HTML escaping utility
 * @param {string} str 
 * @returns {string}
 */
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ============================================================================
// 9. Polling Engine
// ============================================================================
/**
 * Start recurring polling of job status
 * @param {string} jobId 
 * @param {'discovery' | 'extraction'} type 
 * @param {number} interval 
 */
function startPolling(jobId, type, interval = 2000) {
    stopPolling();

    state.activePollingType = type;
    showLoading(true);

    const pollTask = async () => {
        try {
            if (type === 'discovery') {
                const statusData = await api.getDiscoveryStatus(jobId);
                updateDiscoveryProgress(statusData);
            } else if (type === 'extraction') {
                const statusData = await api.getExtractionStatus(jobId);
                updateExtractionProgress(statusData);
            }
        } catch (error) {
            console.error(`Error polling ${type} job ${jobId}:`, error);
            showToast(`خطا در همگام‌سازی وضعیت: ${error.message}`, 'warning');
        }
    };

    // Immediate first tick
    pollTask();

    // Set recurring timer
    state.pollInterval = setInterval(pollTask, interval);
}

/**
 * Stop any active polling interval
 */
function stopPolling() {
    if (state.pollInterval) {
        clearInterval(state.pollInterval);
        state.pollInterval = null;
    }
    state.activePollingType = null;
    showLoading(false);
}

// ============================================================================
// 10. DOMContentLoaded & Event Listeners
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // ------------------------------------------------------------------------
    // Tab switching buttons
    // ------------------------------------------------------------------------
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetTab = btn.getAttribute('data-tab');
            if (targetTab) switchTab(targetTab);
        });
    });

    // ------------------------------------------------------------------------
    // Advanced settings panel toggle
    // ------------------------------------------------------------------------
    const btnToggleAdvanced = document.getElementById('btn-toggle-advanced');
    const advancedPanel = document.getElementById('discovery-advanced-panel');
    if (btnToggleAdvanced && advancedPanel) {
        btnToggleAdvanced.addEventListener('click', () => {
            const isOpen = advancedPanel.classList.toggle('open');
            btnToggleAdvanced.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            advancedPanel.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
        });
    }

    // ------------------------------------------------------------------------
    // Discovery action buttons
    // ------------------------------------------------------------------------
    const btnStartDiscovery = document.getElementById('btn-start-discovery');
    if (btnStartDiscovery) {
        btnStartDiscovery.addEventListener('click', startDiscovery);
    }

    const btnCancelDiscovery = document.getElementById('btn-cancel-discovery');
    if (btnCancelDiscovery) {
        btnCancelDiscovery.addEventListener('click', cancelDiscovery);
    }

    // Root URL Enter key shortcut
    const inputRootUrl = document.getElementById('input-root-url');
    if (inputRootUrl) {
        inputRootUrl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                startDiscovery();
            }
        });
    }

    // ------------------------------------------------------------------------
    // Discovery table actions & search
    // ------------------------------------------------------------------------
    const searchInput = document.getElementById('input-discovery-search');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            filterResults(e.target.value);
        });
    }

    const btnClearSearch = document.getElementById('btn-clear-discovery-search');
    if (btnClearSearch && searchInput) {
        btnClearSearch.addEventListener('click', () => {
            searchInput.value = '';
            filterResults('');
            searchInput.focus();
        });
    }

    const btnSelectAll = document.getElementById('btn-select-all');
    if (btnSelectAll) {
        btnSelectAll.addEventListener('click', selectAll);
    }

    const btnDeselectAll = document.getElementById('btn-deselect-all');
    if (btnDeselectAll) {
        btnDeselectAll.addEventListener('click', deselectAll);
    }

    const headerCheckbox = document.getElementById('header-discovery-checkbox');
    if (headerCheckbox) {
        headerCheckbox.addEventListener('change', (e) => {
            if (e.target.checked) selectAll();
            else deselectAll();
        });
    }

    const btnSendToExtraction = document.getElementById('btn-send-to-extraction');
    if (btnSendToExtraction) {
        btnSendToExtraction.addEventListener('click', sendToExtraction);
    }

    const btnExportDiscovered = document.getElementById('btn-export-discovered-urls');
    if (btnExportDiscovered) {
        btnExportDiscovered.addEventListener('click', exportDiscoveredUrls);
    }

    // ------------------------------------------------------------------------
    // Extraction Tab Controls
    // ------------------------------------------------------------------------
    const btnAddUrls = document.getElementById('btn-add-urls');
    if (btnAddUrls) {
        btnAddUrls.addEventListener('click', addManualExtractionUrls);
    }

    const extractionTextarea = document.getElementById('input-extraction-urls-textarea');
    if (extractionTextarea) {
        extractionTextarea.addEventListener('input', (e) => {
            const count = parseUrlsFromTextarea(e.target.value).length;
            const badge = document.getElementById('urls-input-count');
            if (badge) badge.textContent = `${count} مورد`;
        });
    }

    const btnClearExtractionUrls = document.getElementById('btn-clear-selected-urls');
    if (btnClearExtractionUrls) {
        btnClearExtractionUrls.addEventListener('click', clearAllExtractionUrls);
    }

    const btnStartExtraction = document.getElementById('btn-start-extraction');
    if (btnStartExtraction) {
        btnStartExtraction.addEventListener('click', startExtraction);
    }

    const btnCancelExtraction = document.getElementById('btn-cancel-extraction');
    if (btnCancelExtraction) {
        btnCancelExtraction.addEventListener('click', cancelExtraction);
    }

    const btnRetryFailed = document.getElementById('btn-retry-failed');
    if (btnRetryFailed) {
        btnRetryFailed.addEventListener('click', retryFailedExtraction);
    }

    // Export formats buttons
    document.querySelectorAll('.btn-export').forEach(btn => {
        btn.addEventListener('click', () => {
            const format = btn.getAttribute('data-format');
            if (format) exportResults(format);
        });
    });

    // ------------------------------------------------------------------------
    // Modal dialog controls
    // ------------------------------------------------------------------------
    const btnClosePreview = document.getElementById('btn-close-preview-modal');
    const btnDismissPreview = document.getElementById('btn-dismiss-preview-modal');
    if (btnClosePreview) btnClosePreview.addEventListener('click', closePreviewModal);
    if (btnDismissPreview) btnDismissPreview.addEventListener('click', closePreviewModal);

    const btnCopyPreview = document.getElementById('btn-copy-preview-text');
    if (btnCopyPreview) {
        btnCopyPreview.addEventListener('click', () => {
            const contentEl = document.getElementById('modal-doc-content');
            if (contentEl && contentEl.textContent) {
                navigator.clipboard.writeText(contentEl.textContent)
                    .then(() => showToast('متن مستند در حافظه کپی شد.', 'success'))
                    .catch(() => showToast('خطا در کپی کردن متن.', 'error'));
            }
        });
    }

    const btnCloseError = document.getElementById('btn-close-error-modal');
    const btnDismissError = document.getElementById('btn-dismiss-error-modal');
    if (btnCloseError) btnCloseError.addEventListener('click', closeErrorModal);
    if (btnDismissError) btnDismissError.addEventListener('click', closeErrorModal);

    // Close modals on clicking overlay backdrop
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closePreviewModal();
                closeErrorModal();
            }
        });
    });

    // Escape key closes open modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closePreviewModal();
            closeErrorModal();
        }
    });

    // Initial render
    renderDiscoveryResults([]);
    renderExtractionUrlList();
    renderExtractionResults([]);
    updateSelectedCounter();

    // ============================================================================
    // AI Features Module (LLM RAG · Text Chunking · Vector Embedding)
    // ============================================================================

    // ── AI State ──
    const aiState = {
        apiKeyConfigured: false,
        baseUrl: 'https://api.openai.com/v1',
        llmModel: 'gpt-4o-mini',
        embeddingModel: 'text-embedding-3-small',
        chunksReady: false,
        embeddingsReady: false,
        ragReady: false,
        currentJobId: null,
        lastEnhancedData: null,
        lastChunksData: null,
        activePreviewTab: 'json'
    };

    // ── AI API client methods ──
    const aiApi = {
        async getStatus() {
            const resp = await fetch('/api/ai/status', { headers: { Accept: 'application/json' } });
            if (!resp.ok) throw new Error('Failed to fetch AI status');
            return resp.json();
        },

        async chunk(jobId, strategy, chunkSize, overlap) {
            const resp = await fetch('/api/ai/chunk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
                body: JSON.stringify({ job_id: jobId, strategy, chunk_size: chunkSize, overlap }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.detail || data.message || `Chunk error ${resp.status}`);
            return data;
        },

        async embed(jobId) {
            const resp = await fetch('/api/ai/embed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
                body: JSON.stringify({ job_id: jobId }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.detail || data.message || `Embed error ${resp.status}`);
            return data;
        },

        async enhance(jobId) {
            const resp = await fetch('/api/ai/enhance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
                body: JSON.stringify({ job_id: jobId }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.detail || data.message || `Enhance error ${resp.status}`);
            return data;
        },

        async exportAI(jobId, format) {
            const resp = await fetch('/api/ai/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId, format }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `Export error ${resp.status}`);
            }
            return resp.blob();
        },
    };

    // ── Helper: Set AI Pipeline Step State ──
    function setAIStep(stepId, stepState) {
        const icon = document.getElementById(`ai-step-${stepId}-icon`);
        const status = document.getElementById(`ai-step-${stepId}-status`);
        if (!icon) return;
        icon.className = `ai-step-icon ${stepState.cls}`;
        if (stepState.icon) icon.innerHTML = `<i class="${stepState.icon}"></i>`;
        if (status) status.textContent = stepState.text;
    }

    const AI_STEP_STATES = {
        pending:  (text) => ({ cls: 'pending', text: text || 'در انتظار...' }),
        running:  (text) => ({ cls: 'running', icon: 'fa-solid fa-spinner fa-spin', text: text || 'در حال پردازش...' }),
        done:     (text) => ({ cls: 'done', icon: 'fa-solid fa-check', text: text || 'تکمیل شد ✓' }),
        error:    (text) => ({ cls: 'error', icon: 'fa-solid fa-xmark', text: text || 'خطا' }),
        skipped:  (text) => ({ cls: 'skipped', icon: 'fa-solid fa-minus', text: text || 'رد شد' }),
    };

    function showEl(id, visible) {
        const el = document.getElementById(id);
        if (!el) return;
        if (visible) el.classList.remove('hidden');
        else el.classList.add('hidden');
    }

    // ── Update Studio Info & Button States ──
    function updateAIStudioInfo() {
        const hasJob = !!state.extractionJobId;
        const docCount = state.extractedDocuments.length;

        // Update Job ID and Doc Count in Studio Header
        const jobIdEl = document.getElementById('ai-studio-job-id');
        const docCountEl = document.getElementById('ai-studio-doc-count');
        const tabAiBadge = document.getElementById('ai-badge-count');

        if (jobIdEl) jobIdEl.textContent = state.extractionJobId || 'هیچ کاری اجرا نشده';
        if (docCountEl) docCountEl.textContent = `${docCount.toLocaleString('fa-IR')} سند استخراج‌شده`;
        if (tabAiBadge) {
            tabAiBadge.textContent = docCount > 0 ? `${docCount} سند` : 'AI';
        }

        // Master pipeline run button
        const btnRunAI = document.getElementById('btn-run-ai');
        if (btnRunAI) {
            const doChunk = document.getElementById('check-enable-chunking')?.checked;
            const doEmbed = document.getElementById('check-enable-embedding')?.checked;
            const doRAG = document.getElementById('check-enable-rag')?.checked;
            btnRunAI.disabled = !(hasJob && (doChunk || doEmbed || doRAG));
        }

        // Standalone Buttons
        const btnRunOnlyLLM = document.getElementById('btn-run-only-llm');
        if (btnRunOnlyLLM) {
            btnRunOnlyLLM.disabled = !(hasJob && aiState.apiKeyConfigured);
            btnRunOnlyLLM.title = !hasJob
                ? 'ابتدا باید عملیات استخراج انجام شود'
                : (!aiState.apiKeyConfigured ? 'نیاز به تنظیم CRAWLBOT_OPENAI_API_KEY در فایل .env' : 'اجرای بهبود ساختار با LLM');
        }

        const btnRunOnlyChunk = document.getElementById('btn-run-only-chunking');
        if (btnRunOnlyChunk) {
            btnRunOnlyChunk.disabled = !hasJob;
            btnRunOnlyChunk.title = !hasJob ? 'ابتدا باید عملیات استخراج انجام شود' : 'تقطیع متن اسناد استخراج‌شده';
        }

        const btnRunOnlyEmbed = document.getElementById('btn-run-only-embedding');
        if (btnRunOnlyEmbed) {
            btnRunOnlyEmbed.disabled = !(hasJob && aiState.apiKeyConfigured);
            btnRunOnlyEmbed.title = !hasJob
                ? 'ابتدا باید عملیات استخراج انجام شود'
                : (!aiState.apiKeyConfigured ? 'نیاز به تنظیم CRAWLBOT_OPENAI_API_KEY در فایل .env' : 'تولید بردارهای Embedding');
        }
    }

    // ── Check AI Status on load & populate config ──
    async function checkAIStatus() {
        try {
            const resp = await aiApi.getStatus();
            const d = resp.data || {};
            aiState.apiKeyConfigured = d.api_key_configured || false;
            aiState.baseUrl = d.base_url || 'https://api.openai.com/v1';
            aiState.llmModel = d.llm_model || 'gpt-4o-mini';
            aiState.embeddingModel = d.embedding_model || 'text-embedding-3-small';

            showEl('ai-status-checking', false);
            if (aiState.apiKeyConfigured) {
                showEl('ai-status-ready', true);
                showEl('ai-status-no-key', false);
            } else {
                showEl('ai-status-ready', false);
                showEl('ai-status-no-key', true);
                const embedCheck = document.getElementById('check-enable-embedding');
                const ragCheck = document.getElementById('check-enable-rag');
                if (embedCheck) { embedCheck.disabled = true; embedCheck.title = 'نیاز به CRAWLBOT_OPENAI_API_KEY'; }
                if (ragCheck) { ragCheck.disabled = true; ragCheck.title = 'نیاز به CRAWLBOT_OPENAI_API_KEY'; }
            }

            // Populate labels in UI
            const baseUrlEl = document.getElementById('ai-status-base-url');
            if (baseUrlEl) baseUrlEl.textContent = aiState.baseUrl;

            const llmStatusEl = document.getElementById('ai-status-llm-model');
            if (llmStatusEl) llmStatusEl.textContent = aiState.llmModel;
            const cardLlmLabel = document.getElementById('card-llm-model-label');
            if (cardLlmLabel) cardLlmLabel.textContent = aiState.llmModel;

            const embedStatusEl = document.getElementById('ai-status-embed-model');
            if (embedStatusEl) embedStatusEl.textContent = aiState.embeddingModel;
            const cardEmbedLabel = document.getElementById('card-embed-model-label');
            if (cardEmbedLabel) cardEmbedLabel.textContent = aiState.embeddingModel;
            const badgeEmbedName = document.getElementById('badge-embed-model-name');
            if (badgeEmbedName) badgeEmbedName.textContent = aiState.embeddingModel;

        } catch (err) {
            console.error('Failed to check AI status:', err);
            showEl('ai-status-checking', false);
            showEl('ai-status-no-key', true);
        } finally {
            updateAIStudioInfo();
        }
    }

    // ── Live Preview Renderer ──
    function updateLivePreview(tab) {
        if (tab) aiState.activePreviewTab = tab;
        const codeEl = document.getElementById('ai-preview-code');
        const tagEl = document.getElementById('ai-preview-status-tag');
        const tabJson = document.getElementById('tab-preview-json');
        const tabChunks = document.getElementById('tab-preview-chunks');

        if (tabJson && tabChunks) {
            tabJson.classList.toggle('active', aiState.activePreviewTab === 'json');
            tabChunks.classList.toggle('active', aiState.activePreviewTab === 'chunks');
        }

        if (!codeEl) return;

        if (aiState.activePreviewTab === 'json') {
            if (aiState.lastEnhancedData) {
                codeEl.textContent = JSON.stringify(aiState.lastEnhancedData, null, 2);
                if (tagEl) tagEl.textContent = 'نمایش خروجی ساختاریافته RAG (خلاصه‌ها، کلیدواژه‌ها و Q&A)';
            } else if (state.extractedDocuments.length > 0) {
                const sample = state.extractedDocuments.slice(0, 2);
                codeEl.textContent = JSON.stringify({
                    info: "اسناد استخراج‌شده آماده هستند. برای تولید ساختار هوشمند RAG، دکمه «اجرای این مرحله با LLM» را بزنید.",
                    sample_documents: sample
                }, null, 2);
                if (tagEl) tagEl.textContent = 'پیش‌نمایش اسناد اولیه (قبل از بهبود با LLM)';
            } else {
                codeEl.textContent = '// هنوز مستندی استخراج نشده است. ابتدا در تب استخراج، صفحات را دریافت نمایید.';
                if (tagEl) tagEl.textContent = 'در انتظار اجرای استخراج';
            }
        } else { // 'chunks'
            if (aiState.lastChunksData) {
                const chunksList = Array.isArray(aiState.lastChunksData)
                    ? aiState.lastChunksData.slice(0, 5)
                    : (aiState.lastChunksData.chunks ? aiState.lastChunksData.chunks.slice(0, 5) : aiState.lastChunksData);
                codeEl.textContent = JSON.stringify(chunksList, null, 2);
                if (tagEl) tagEl.textContent = 'نمایش نمونه ۵ قطعه اول تقطیع‌شده (Chunks) آماده RAG';
            } else {
                codeEl.textContent = '// هنوز تقطیع انجام نشده است. روی دکمه «اجرای تقطیع (Chunking)» کلیک فرمایید.';
                if (tagEl) tagEl.textContent = 'در انتظار تقطیع';
            }
        }
    }

    // Preview Tab button clicks
    document.getElementById('tab-preview-json')?.addEventListener('click', () => updateLivePreview('json'));
    document.getElementById('tab-preview-chunks')?.addEventListener('click', () => updateLivePreview('chunks'));

    // Navigation Shortcuts
    document.getElementById('btn-goto-ai-tab')?.addEventListener('click', () => {
        switchTab('ai');
        checkAIStatus();
    });

    document.getElementById('btn-goto-ai-from-results')?.addEventListener('click', () => {
        switchTab('ai');
        checkAIStatus();
    });

    // ── AI Export Handler ──
    async function handleAIExport(format, filename) {
        const jobId = aiState.currentJobId || state.extractionJobId;
        if (!jobId) {
            showToast('شناسه کار استخراج یافت نشد.', 'error');
            return;
        }
        try {
            showToast(`در حال آماده‌سازی فایل خروجی ${filename}...`, 'info');
            const blob = await aiApi.exportAI(jobId, format);
            triggerBlobDownload(blob, filename);
            showToast(`فایل ${filename} با موفقیت دانلود شد.`, 'success');
        } catch (err) {
            showToast(`خطا در دانلود فایل: ${err.message}`, 'error');
        }
    }

    // Attach Export Buttons
    document.getElementById('btn-export-rag-json')?.addEventListener('click', () => {
        const jId = (aiState.currentJobId || state.extractionJobId || 'job').slice(0, 8);
        handleAIExport('json_enhanced', `rag_enhanced_${jId}.json`);
    });
    document.getElementById('btn-export-rag-jsonl')?.addEventListener('click', () => {
        const jId = (aiState.currentJobId || state.extractionJobId || 'job').slice(0, 8);
        handleAIExport('jsonl_enhanced', `rag_enhanced_${jId}.jsonl`);
    });
    document.getElementById('btn-export-rag-markdown')?.addEventListener('click', () => {
        const jId = (aiState.currentJobId || state.extractionJobId || 'job').slice(0, 8);
        handleAIExport('markdown_enhanced', `rag_markdown_${jId}.zip`);
    });
    document.getElementById('btn-export-chunks-jsonl')?.addEventListener('click', () => {
        const jId = (aiState.currentJobId || state.extractionJobId || 'job').slice(0, 8);
        handleAIExport('jsonl_chunks', `chunks_${jId}.jsonl`);
    });
    document.getElementById('btn-export-chunks-json')?.addEventListener('click', () => {
        const jId = (aiState.currentJobId || state.extractionJobId || 'job').slice(0, 8);
        handleAIExport('json_chunks', `chunks_${jId}.json`);
    });
    document.getElementById('btn-export-chunks-vectors')?.addEventListener('click', () => {
        const jId = (aiState.currentJobId || state.extractionJobId || 'job').slice(0, 8);
        handleAIExport('jsonl_chunks', `chunks_embedded_${jId}.jsonl`);
    });

    // ── Standalone Feature 1: LLM RAG Structuring ──
    const btnRunOnlyLLM = document.getElementById('btn-run-only-llm');
    if (btnRunOnlyLLM) {
        btnRunOnlyLLM.addEventListener('click', async () => {
            const jobId = state.extractionJobId;
            if (!jobId) {
                showToast('ابتدا باید استخراج صفحات انجام شود.', 'error');
                return;
            }
            if (!aiState.apiKeyConfigured) {
                showToast('کلید CRAWLBOT_OPENAI_API_KEY در .env تنظیم نشده است.', 'warning');
                return;
            }

            const origContent = btnRunOnlyLLM.innerHTML;
            btnRunOnlyLLM.disabled = true;
            btnRunOnlyLLM.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>در حال پردازش با LLM...</span>';
            showToast('پردازش ساختار دقیق RAG با مدل LLM آغاز شد...', 'info');

            try {
                const res = await aiApi.enhance(jobId);
                aiState.currentJobId = jobId;
                aiState.ragReady = true;
                aiState.lastEnhancedData = res.data;

                const count = res.data?.enhanced_docs || state.extractedDocuments.length;
                document.getElementById('ai-stat-enhanced-value').textContent = count;
                showEl('ai-stats-panel', true);

                // Enable RAG export buttons
                document.getElementById('btn-export-rag-json').disabled = false;
                document.getElementById('btn-export-rag-jsonl').disabled = false;
                document.getElementById('btn-export-rag-markdown').disabled = false;

                // Update Preview to JSON tab
                updateLivePreview('json');
                showToast(`پردازش LLM تکمیل شد: ${count} سند با ساختار استاندارد RAG غنی‌سازی شدند.`, 'success');
            } catch (err) {
                showToast(`خطا در پردازش LLM: ${err.message}`, 'error');
            } finally {
                btnRunOnlyLLM.disabled = false;
                btnRunOnlyLLM.innerHTML = origContent;
                updateAIStudioInfo();
            }
        });
    }

    // ── Standalone Feature 2: Text Chunking ──
    const btnRunOnlyChunking = document.getElementById('btn-run-only-chunking');
    if (btnRunOnlyChunking) {
        btnRunOnlyChunking.addEventListener('click', async () => {
            const jobId = state.extractionJobId;
            if (!jobId) {
                showToast('ابتدا باید استخراج صفحات انجام شود.', 'error');
                return;
            }

            const strategy = document.getElementById('select-chunk-strategy')?.value || 'auto';
            const chunkSize = parseInt(document.getElementById('input-chunk-size')?.value || '1000', 10);
            const overlap = parseInt(document.getElementById('input-chunk-overlap')?.value || '150', 10);

            const origContent = btnRunOnlyChunking.innerHTML;
            btnRunOnlyChunking.disabled = true;
            btnRunOnlyChunking.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>در حال تقطیع متن...</span>';
            showToast('عملیات تقطیع متن (Chunking) آغاز شد...', 'info');

            try {
                const res = await aiApi.chunk(jobId, strategy, chunkSize, overlap);
                aiState.currentJobId = jobId;
                aiState.chunksReady = true;
                aiState.lastChunksData = res.data;

                const count = res.data?.total_chunks || 0;
                document.getElementById('ai-stat-chunks-value').textContent = count;
                showEl('ai-stats-panel', true);

                // Enable chunk export buttons
                document.getElementById('btn-export-chunks-jsonl').disabled = false;
                document.getElementById('btn-export-chunks-json').disabled = false;

                // Update Preview to chunks tab
                updateLivePreview('chunks');
                const stratName = res.data?.applied_strategy ? `${res.data.strategy} (${res.data.applied_strategy})` : (res.data?.strategy || strategy);
                showToast(`تقطیع با موفقیت انجام شد: ${count} قطعه (Chunk) با استراتژی ${stratName} ایجاد گردید.`, 'success');
            } catch (err) {
                showToast(`خطا در تقطیع متن: ${err.message}`, 'error');
            } finally {
                btnRunOnlyChunking.disabled = false;
                btnRunOnlyChunking.innerHTML = origContent;
                updateAIStudioInfo();
            }
        });
    }

    // ── Standalone Feature 3: Vector Embedding ──
    const btnRunOnlyEmbedding = document.getElementById('btn-run-only-embedding');
    if (btnRunOnlyEmbedding) {
        btnRunOnlyEmbedding.addEventListener('click', async () => {
            const jobId = state.extractionJobId;
            if (!jobId) {
                showToast('ابتدا باید استخراج صفحات انجام شود.', 'error');
                return;
            }
            if (!aiState.apiKeyConfigured) {
                showToast('کلید CRAWLBOT_OPENAI_API_KEY در .env تنظیم نشده است.', 'warning');
                return;
            }

            const origContent = btnRunOnlyEmbedding.innerHTML;
            btnRunOnlyEmbedding.disabled = true;
            btnRunOnlyEmbedding.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>در حال وکتورسازی...</span>';

            try {
                // If chunks haven't been generated yet, do chunking first automatically
                if (!aiState.chunksReady) {
                    showToast('ابتدا در حال تقطیع خودکار متن برای ایجاد چانک‌ها...', 'info');
                    const strategy = document.getElementById('select-chunk-strategy')?.value || 'auto';
                    const chunkSize = parseInt(document.getElementById('input-chunk-size')?.value || '1000', 10);
                    const overlap = parseInt(document.getElementById('input-chunk-overlap')?.value || '150', 10);
                    const chunkRes = await aiApi.chunk(jobId, strategy, chunkSize, overlap);
                    aiState.chunksReady = true;
                    aiState.lastChunksData = chunkRes.data;
                    document.getElementById('ai-stat-chunks-value').textContent = chunkRes.data?.total_chunks || 0;
                    document.getElementById('btn-export-chunks-jsonl').disabled = false;
                    document.getElementById('btn-export-chunks-json').disabled = false;
                }

                showToast(`ارسال دسته‌ای چانک‌ها به مدل ${aiState.embeddingModel}...`, 'info');
                const embedRes = await aiApi.embed(jobId);
                aiState.currentJobId = jobId;
                aiState.embeddingsReady = true;

                const embeddedCount = embedRes.data?.embedded_chunks || 0;
                document.getElementById('ai-stat-embedded-value').textContent = embeddedCount;
                showEl('ai-stats-panel', true);

                // Enable vector export button
                document.getElementById('btn-export-chunks-vectors').disabled = false;

                // Update Preview
                updateLivePreview('chunks');
                showToast(`تولید بردارها با موفقیت پایان یافت: ${embeddedCount} بردار با ابعاد ۱۵۳۶ تولید شد.`, 'success');
            } catch (err) {
                showToast(`خطا در تولید بردارهای Embedding: ${err.message}`, 'error');
            } finally {
                btnRunOnlyEmbedding.disabled = false;
                btnRunOnlyEmbedding.innerHTML = origContent;
                updateAIStudioInfo();
            }
        });
    }

    // ── Master All-in-One Pipeline Execution ──
    const btnRunAI = document.getElementById('btn-run-ai');
    if (btnRunAI) {
        btnRunAI.addEventListener('click', async () => {
            const jobId = state.extractionJobId;
            if (!jobId) {
                showToast('ابتدا باید استخراج صفحات انجام شود.', 'error');
                return;
            }

            const doChunk = document.getElementById('check-enable-chunking')?.checked;
            const doEmbed = document.getElementById('check-enable-embedding')?.checked;
            const doRAG = document.getElementById('check-enable-rag')?.checked;

            if (!doChunk && !doEmbed && !doRAG) {
                showToast('حداقل یک قابلیت هوش مصنوعی را انتخاب کنید.', 'warning');
                return;
            }

            showEl('ai-progress-section', true);
            btnRunAI.disabled = true;
            aiState.currentJobId = jobId;

            ['rag', 'chunking', 'embedding'].forEach(s => setAIStep(s, AI_STEP_STATES.pending()));

            let enhancedCount = 0;
            let chunkCount = 0;
            let embeddedCount = 0;

            // ─ Step 1: LLM RAG Structuring ─
            if (doRAG) {
                setAIStep('rag', AI_STEP_STATES.running('در حال پردازش ساختار با LLM...'));
                try {
                    const result = await aiApi.enhance(jobId);
                    enhancedCount = result.data?.enhanced_docs || state.extractedDocuments.length;
                    aiState.ragReady = true;
                    aiState.lastEnhancedData = result.data;
                    setAIStep('rag', AI_STEP_STATES.done(`${enhancedCount} سند بهبود یافت ✓`));
                    document.getElementById('btn-export-rag-json').disabled = false;
                    document.getElementById('btn-export-rag-jsonl').disabled = false;
                    document.getElementById('btn-export-rag-markdown').disabled = false;
                } catch (err) {
                    setAIStep('rag', AI_STEP_STATES.error(err.message));
                    showToast(`خطا در بهبود RAG: ${err.message}`, 'error');
                }
            } else {
                setAIStep('rag', AI_STEP_STATES.skipped('انتخاب نشده'));
            }

            // ─ Step 2: Chunking ─
            if (doChunk) {
                setAIStep('chunking', AI_STEP_STATES.running('در حال تقطیع متن...'));
                try {
                    const strategy = document.getElementById('select-chunk-strategy')?.value || 'auto';
                    const chunkSize = parseInt(document.getElementById('input-chunk-size')?.value || '1000', 10);
                    const overlap = parseInt(document.getElementById('input-chunk-overlap')?.value || '150', 10);
                    const result = await aiApi.chunk(jobId, strategy, chunkSize, overlap);
                    chunkCount = result.data?.total_chunks || 0;
                    aiState.chunksReady = true;
                    aiState.lastChunksData = result.data;
                    const stratDesc = result.data?.applied_strategy ? ` [${result.data.applied_strategy}]` : '';
                    setAIStep('chunking', AI_STEP_STATES.done(`${chunkCount} قطعه تولید شد${stratDesc} ✓`));
                    document.getElementById('btn-export-chunks-jsonl').disabled = false;
                    document.getElementById('btn-export-chunks-json').disabled = false;
                } catch (err) {
                    setAIStep('chunking', AI_STEP_STATES.error(err.message));
                    showToast(`خطا در تقطیع: ${err.message}`, 'error');
                }
            } else {
                setAIStep('chunking', AI_STEP_STATES.skipped('انتخاب نشده'));
            }

            // ─ Step 3: Embedding ─
            if (doEmbed) {
                if (!aiState.chunksReady && !doChunk) {
                    setAIStep('embedding', AI_STEP_STATES.error('ابتدا تقطیع (Chunking) را فعال کنید'));
                    showToast('برای وکتورسازی ابتدا باید تقطیع انجام شود.', 'warning');
                } else {
                    setAIStep('embedding', AI_STEP_STATES.running('در حال تولید بردارهای Embedding...'));
                    try {
                        const result = await aiApi.embed(jobId);
                        embeddedCount = result.data?.embedded_chunks || 0;
                        aiState.embeddingsReady = true;
                        setAIStep('embedding', AI_STEP_STATES.done(`${embeddedCount} بردار تولید شد ✓`));
                        document.getElementById('btn-export-chunks-vectors').disabled = false;
                    } catch (err) {
                        setAIStep('embedding', AI_STEP_STATES.error(err.message));
                        showToast(`خطا در وکتورسازی: ${err.message}`, 'error');
                    }
                }
            } else {
                setAIStep('embedding', AI_STEP_STATES.skipped('انتخاب نشده'));
            }

            // Show Stats
            showEl('ai-stats-panel', true);
            document.getElementById('ai-stat-enhanced-value').textContent = doRAG ? enhancedCount : '—';
            document.getElementById('ai-stat-chunks-value').textContent = doChunk ? chunkCount : '—';
            document.getElementById('ai-stat-embedded-value').textContent = doEmbed ? embeddedCount : '—';

            // Update live preview
            updateLivePreview(doRAG ? 'json' : (doChunk ? 'chunks' : 'json'));

            btnRunAI.disabled = false;
            updateAIStudioInfo();
            showToast('عملیات فرآیند هوش مصنوعی به پایان رسید.', 'success');
        });
    }

    // Checkbox changes re-evaluate master run button
    ['check-enable-chunking', 'check-enable-embedding', 'check-enable-rag'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', updateAIStudioInfo);
    });

    // ------------------------------------------------------------------------
    // Module: File & Data Processing Studio
    // ------------------------------------------------------------------------
    const fileState = {
        uploadedFileId: null,
        uploadedFileInfo: null,
        lastArtifactData: null,
        activePreviewTab: 'output',
    };

    const filesApi = {
        async upload(file) {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch('/api/files/upload', {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || 'خطا در بارگذاری فایل');
            }
            return res.json();
        },
        async process(fileId, operations, options) {
            const res = await fetch('/api/files/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_id: fileId, operations, options }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || 'خطا در پردازش فایل');
            }
            return res.json();
        },
    };

    function initFileProcessingStudio() {
        const dropzone = document.getElementById('file-dropzone');
        const fileInput = document.getElementById('input-file-upload');
        const btnRunProcess = document.getElementById('btn-run-file-process');
        const fileInfoPanel = document.getElementById('file-info-panel');
        const fileResultsPanel = document.getElementById('file-results-panel');

        if (!dropzone || !fileInput) return;

        dropzone.addEventListener('click', () => fileInput.click());

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer?.files;
            if (files && files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleFileSelect(fileInput.files[0]);
            }
        });

        async function handleFileSelect(file) {
            showToast(`در حال بارگذاری فایل ${file.name}...`, 'info');
            try {
                const res = await filesApi.upload(file);
                const info = res.data;
                fileState.uploadedFileId = info.file_id;
                fileState.uploadedFileInfo = info;

                document.getElementById('file-info-name').textContent = info.filename;
                document.getElementById('file-info-format').textContent = (info.format || '').toUpperCase();
                document.getElementById('file-info-size').textContent = `${(info.size_bytes / 1024).toFixed(1)} KB`;
                document.getElementById('file-info-lines').textContent = info.line_count || '0';
                document.getElementById('file-info-encoding').textContent = info.encoding || 'UTF-8';

                showEl('file-info-panel', true);
                if (btnRunProcess) btnRunProcess.disabled = false;
                const badge = document.getElementById('files-badge-count');
                if (badge) badge.textContent = '1';

                showToast(`فایل ${info.filename} با موفقیت بارگذاری گردید.`, 'success');
            } catch (err) {
                showToast(`خطا در بارگذاری فایل: ${err.message}`, 'error');
            }
        }

        if (btnRunProcess) {
            btnRunProcess.addEventListener('click', async () => {
                if (!fileState.uploadedFileId) {
                    showToast('ابتدا باید یک فایل معتبر بارگذاری نمایید.', 'warning');
                    return;
                }

                const ops = [];
                if (document.getElementById('check-file-op-normalize')?.checked) ops.push('normalize');
                if (document.getElementById('check-file-op-md2json')?.checked) ops.push('markdown_to_json');
                if (document.getElementById('check-file-op-validate')?.checked) ops.push('json_validate');
                if (document.getElementById('check-file-op-cleanup')?.checked) ops.push('json_cleanup');
                if (document.getElementById('check-file-op-chunk')?.checked) ops.push('chunk');
                if (document.getElementById('check-file-op-enrich')?.checked) ops.push('metadata_enrich');

                if (ops.length === 0) {
                    showToast('حداقل یک عملیات پردازشی را انتخاب نمایید.', 'warning');
                    return;
                }

                const options = {
                    chunk_strategy: document.getElementById('select-file-chunk-strategy')?.value || 'auto',
                    chunk_size: parseInt(document.getElementById('input-file-chunk-size')?.value || '1000', 10),
                    overlap: parseInt(document.getElementById('input-file-chunk-overlap')?.value || '150', 10),
                    category: document.getElementById('input-file-category')?.value || null,
                    output_format: document.getElementById('select-file-output-format')?.value || 'json',
                };

                const origHtml = btnRunProcess.innerHTML;
                btnRunProcess.disabled = true;
                btnRunProcess.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>در حال پردازش خط لوله...</span>';
                showToast('فرآیند پردازش خط لوله آغاز شد...', 'info');

                try {
                    const res = await filesApi.process(fileState.uploadedFileId, ops, options);
                    const data = res.data;
                    fileState.lastArtifactData = data;

                    document.getElementById('stat-file-records-in').textContent = data.telemetry?.records_in || 0;
                    document.getElementById('stat-file-records-out').textContent = data.telemetry?.records_out || 0;
                    document.getElementById('stat-file-chunks').textContent = data.telemetry?.chunks_produced || 0;
                    document.getElementById('stat-file-fixes').textContent = data.telemetry?.normalization_fixes || 0;
                    document.getElementById('stat-file-duration').textContent = `${data.duration_seconds || 0}s`;

                    const btnDownload = document.getElementById('btn-download-file-result');
                    if (btnDownload) {
                        btnDownload.href = data.download_url;
                        btnDownload.setAttribute('download', data.filename);
                    }

                    const preOutput = document.getElementById('file-preview-output-content');
                    if (preOutput) preOutput.textContent = data.output_preview || '// خروجی خالی';

                    const preDiff = document.getElementById('file-preview-diff-content');
                    if (preDiff) preDiff.textContent = data.diff || '// تغییری نسبت به نسخه اولیه وجود ندارد.';

                    showEl('file-results-panel', true);
                    showToast('پردازش خط لوله فایل با موفقیت کامل انجام شد.', 'success');
                } catch (err) {
                    showToast(`خطا در پردازش فایل: ${err.message}`, 'error');
                } finally {
                    btnRunProcess.disabled = false;
                    btnRunProcess.innerHTML = origHtml;
                }
            });
        }

        const tabOutput = document.getElementById('tab-file-preview-output');
        const tabDiff = document.getElementById('tab-file-preview-diff');
        const codeOutput = document.getElementById('file-preview-output-content');
        const codeDiff = document.getElementById('file-preview-diff-content');

        if (tabOutput && tabDiff) {
            tabOutput.addEventListener('click', () => {
                tabOutput.classList.add('active');
                tabDiff.classList.remove('active');
                if (codeOutput) codeOutput.classList.remove('hidden');
                if (codeDiff) codeDiff.classList.add('hidden');
            });
            tabDiff.addEventListener('click', () => {
                tabDiff.classList.add('active');
                tabOutput.classList.remove('active');
                if (codeOutput) codeOutput.classList.add('hidden');
                if (codeDiff) codeDiff.classList.remove('hidden');
            });
        }
    }

    // Initial check on load
    checkAIStatus();
    updateLivePreview('json');
    initFileProcessingStudio();

    // Periodic synchronization with active extraction job
    setInterval(() => {
        updateAIStudioInfo();
    }, 1500);

});
