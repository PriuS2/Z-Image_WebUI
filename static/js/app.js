// Z-Image WebUI - JavaScript (다중 사용자 지원)

// ============= 전역 변수 =============
let ws = null;
let isGenerating = false;
let isModelLoading = false;
let templates = {};
let isTranslating = false;
let isLlmProcessing = false;  // LLM 처리 중 여부
let lastHistoryId = null;
let isAdmin = false;  // 관리자 여부
let sessionId = null;  // 현재 세션 ID
let pendingEditQuantizationValue = null; // 설정에서 내려온 편집 양자화(옵션 로드 전 임시 보관)

// ============= 관리자 GPU 설정/모니터링 =============
let adminGpuSettings = {
    generation_gpu: 'auto',
    edit_gpu: 'auto'
};
let adminAvailableDevices = ['auto', 'cpu'];

// ============= DOM 요소 =============
const chatMessages = document.getElementById('chatMessages');
const promptInput = document.getElementById('promptInput');
const koreanInput = document.getElementById('koreanInput');
const modelStatus = document.getElementById('modelStatus');

// ============= WebSocket 연결 =============
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    
    ws.onopen = () => {
        console.log('WebSocket 연결됨');
        // 핑 전송 시작 (연결 유지)
        startPing();
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onclose = () => {
        console.log('WebSocket 연결 끊김, 재연결 시도...');
        stopPing();
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket 오류:', error);
    };
}

let pingInterval = null;

function startPing() {
    pingInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);  // 30초마다 핑
}

function stopPing() {
    if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
    }
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'connected':
            addMessage('system', data.content);
            if (data.session_id) {
                sessionId = data.session_id;
                console.log('세션 ID:', sessionId);
            }
            if (data.connected_users) {
                updateUserCount(data.connected_users);
            }
            break;
            
        case 'system':
        case 'warning':
            addMessage('system', data.content);
            updateProgressFromMessage(data.content);
            break;
            
        case 'progress':
            addMessage('system', data.content);
            updateProgressFromMessage(data.content);
            break;
            
        case 'image_progress':
            showProgress(`이미지 ${data.current}/${data.total} - 준비 중`, data.progress);
            break;

        case 'generation_progress':
            // 대화 탭 생성 진행 상황 (이미지 n/N - 스텝 n/N)
            handleGenerationProgress(data);
            break;
            
        case 'model_progress':
            updateModelProgress(data.progress, data.label, data.detail, data.stage || '');
            setModelLoadingState(data.stage !== 'complete' && data.stage !== 'error');
            break;
            
        case 'model_status_change':
            // 모델 상태 변경 (모든 사용자에게 동기화)
            updateModelStatusFromData(data);
            break;
            
        case 'complete':
            addMessage('system', data.content);
            updateModelStatus();
            hideProgress();
            break;
            
        case 'error':
            addMessage('system', data.content, 'error');
            hideProgress();
            hideQueueStatus();
            isGenerating = false;
            setGenerateButtonState(false);
            break;
            
        case 'queue_status':
            handleQueueStatus(data);
            break;
            
        case 'queue_update':
            // 큐 상태 전체 업데이트 (다른 사용자 포함)
            // 필요시 UI 업데이트
            break;
            
        case 'generation_result':
            // 이미지 생성 결과
            handleGenerationResult(data);
            break;
            
        case 'user_count':
            updateUserCount(data.count);
            break;
            
        case 'pong':
            // 핑 응답 (무시)
            break;
        
        // ============= 편집 모델 관련 메시지 =============
        case 'edit_model_progress':
            updateEditProgress(data.progress, data.label, data.detail);
            if (data.stage === 'complete' || data.stage === 'error') {
                setTimeout(hideEditProgress, 1500);
                setEditModelLoadingState(false);
            }
            break;
        
        case 'edit_model_status_change':
            updateEditModelStatusFromData(data);
            break;
        
        case 'edit_progress':
            // 편집 진행 상황
            handleEditProgress(data);
            break;
        
        case 'edit_system':
            // 편집 탭 시스템 메시지
            addEditMessage('system', data.content);
            break;
        
        case 'edit_result':
            // 편집 결과
            handleEditResult(data);
            break;
    }
}

// ============= 생성(대화 탭) 진행 상황 처리 =============
function handleGenerationProgress(data) {
    const { current_image, total_images, current_step, total_steps, progress } = data;
    
    let label;
    if (total_images > 1) {
        label = `이미지 ${current_image}/${total_images} - 스텝 ${current_step}/${total_steps}`;
    } else {
        label = `스텝 ${current_step}/${total_steps}`;
    }
    
    showProgress(label, progress);
}


// ============= 편집 진행 상황 처리 =============
function handleEditProgress(data) {
    const { current_image, total_images, current_step, total_steps, progress } = data;
    
    let label;
    if (total_images > 1) {
        label = `이미지 ${current_image}/${total_images} - 스텝 ${current_step}/${total_steps}`;
    } else {
        label = `스텝 ${current_step}/${total_steps}`;
    }
    
    showEditProgress(label, progress);
}


// ============= 편집 결과 처리 =============
function handleEditResult(data) {
    if (data.images && data.images.length > 0) {
        // 원본 이미지 src 가져오기
        const originalImg = document.getElementById('editPreviewImage');
        const originalSrc = originalImg ? originalImg.src : '';
        
        addEditImageMessage(originalSrc, data.images, data.prompt);
    }
    
    hideEditProgress();
    isEditing = false;
    setEditButtonState(false);
}

// ============= 큐 상태 처리 =============
function handleQueueStatus(data) {
    const queueStatus = document.getElementById('queueStatus');
    const queueStatusText = document.getElementById('queueStatusText');
    const queueStatusPosition = document.getElementById('queueStatusPosition');
    
    if (data.status === 'queued' || data.status === 'waiting') {
        queueStatus.style.display = 'flex';
        queueStatusText.textContent = data.message || '대기 중...';
        queueStatusPosition.textContent = `순서: ${data.position}`;
        queueStatus.classList.remove('processing');
        queueStatus.classList.add('waiting');
    } else if (data.status === 'processing') {
        queueStatus.style.display = 'flex';
        queueStatusText.textContent = data.message || '생성 중...';
        queueStatusPosition.textContent = '';
        queueStatus.classList.remove('waiting');
        queueStatus.classList.add('processing');
    } else {
        hideQueueStatus();
    }
}

function hideQueueStatus() {
    const queueStatus = document.getElementById('queueStatus');
    if (queueStatus) {
        queueStatus.style.display = 'none';
    }
}

// ============= 생성 결과 처리 =============
function handleGenerationResult(data) {
    if (data.images && data.images.length > 0) {
        addImageMessage(data.images, data.prompt);
        
        if (data.history_id) {
            lastHistoryId = data.history_id;
            setTimeout(() => {
                saveConversationToHistory(data.history_id);
            }, 500);
        }
    }
    
    hideQueueStatus();
    hideProgress();
    isGenerating = false;
    setGenerateButtonState(false);
}

// ============= 접속자 수 업데이트 =============
function updateUserCount(count) {
    const userCountText = document.getElementById('userCountText');
    if (userCountText) {
        userCountText.textContent = count;
    }
}

// ============= 모델 상태 동기화 =============
function updateModelStatusFromData(data) {
    const indicator = modelStatus.querySelector('.status-indicator');
    const text = modelStatus.querySelector('span');
    const statusBadge = document.getElementById('modelStatusBadge');
    const dot = statusBadge?.querySelector('.status-dot');
    const badgeText = statusBadge?.querySelector('.status-text');
    
    if (data.model_loaded) {
        indicator.classList.add('online');
        indicator.classList.remove('offline');
        text.textContent = '모델 로드됨';
        
        if (dot) {
            dot.classList.remove('offline', 'loading');
            dot.classList.add('online');
        }
        if (badgeText && data.current_model) {
            badgeText.textContent = `✓ ${data.current_model.split(' ')[0]}`;
        }
    } else {
        indicator.classList.remove('online');
        indicator.classList.add('offline');
        text.textContent = '모델 미로드';
        
        if (dot) {
            dot.classList.remove('online', 'loading');
            dot.classList.add('offline');
        }
        if (badgeText) badgeText.textContent = '모델 미로드';
    }
}

// ============= 프로그레스 바 관리 =============
let currentStage = '';

function showProgress(label = '작업 중...', percent = 0, stage = '') {
    const container = document.getElementById('progressContainer');
    const labelEl = document.getElementById('progressLabel');
    const percentEl = document.getElementById('progressPercent');
    const fillEl = document.getElementById('progressFill');
    
    container.style.display = 'block';
    labelEl.textContent = label;
    percentEl.textContent = `${Math.round(percent)}%`;
    fillEl.style.width = `${percent}%`;
    
    if (stage) {
        currentStage = stage;
        fillEl.className = 'progress-fill';
        if (stage === 'download') {
            fillEl.classList.add('downloading');
        } else if (stage === 'error') {
            fillEl.classList.add('error');
        } else if (stage === 'complete') {
            fillEl.classList.add('complete');
        }
    }
}

function updateProgress(percent, label = null, detail = null, stage = null) {
    const labelEl = document.getElementById('progressLabel');
    const percentEl = document.getElementById('progressPercent');
    const fillEl = document.getElementById('progressFill');
    const detailEl = document.getElementById('progressDetail');
    
    if (label) labelEl.textContent = label;
    percentEl.textContent = `${Math.round(percent)}%`;
    fillEl.style.width = `${percent}%`;
    if (detail) detailEl.textContent = detail;
    
    if (stage) {
        currentStage = stage;
        fillEl.className = 'progress-fill';
        if (stage === 'download') {
            fillEl.classList.add('downloading');
        } else if (stage === 'error') {
            fillEl.classList.add('error');
        } else if (stage === 'complete') {
            fillEl.classList.add('complete');
        }
    }
}

function hideProgress() {
    const container = document.getElementById('progressContainer');
    container.style.display = 'none';
    document.getElementById('progressDetail').textContent = '';
    currentStage = '';
}

function updateModelProgress(progress, label, detail, stage = '') {
    showProgress(label, progress, stage);
    if (detail) {
        document.getElementById('progressDetail').textContent = detail;
    }
}

function updateProgressFromMessage(message) {
    if (message.includes('모델 로딩 중')) {
        showProgress('모델 로딩 중...', 30);
        setModelLoadingState(true);
    } else if (message.includes('다운로드')) {
        showProgress('모델 다운로드 중...', 10);
    } else if (message.includes('모델 로드 완료')) {
        updateProgress(100, '모델 로드 완료!');
        setTimeout(hideProgress, 1500);
        setModelLoadingState(false);
    } else if (message.includes('모델 언로드 완료')) {
        hideProgress();
        setModelLoadingState(false);
    } else if (message.includes('이미지 생성 중')) {
        const match = message.match(/\((\d+)\/(\d+)\)/);
        if (match) {
            const current = parseInt(match[1]);
            const total = parseInt(match[2]);
            const percent = (current / total) * 100;
            showProgress(`이미지 ${current}/${total} - 준비 중`, percent);
        }
    } else if (message.includes('생성 완료')) {
        updateProgress(100, '생성 완료!');
        setTimeout(hideProgress, 1000);
    }
}

// ============= 메시지 표시 =============
function addMessage(type, content, style = '') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type} ${style}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<p>${content}</p>`;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function addImageMessage(images, prompt) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const imagesDiv = document.createElement('div');
    imagesDiv.className = 'message-images';
    
    // 이미지 목록 생성 (네비게이션용)
    const imageList = images.map(img => ({
        path: img.path,
        metadata: { prompt, seed: img.seed, width: img.width, height: img.height }
    }));
    
    images.forEach((img, index) => {
        const imgEl = document.createElement('img');
        // base64가 있으면 사용, 없으면 path 사용
        imgEl.src = img.base64 ? `data:image/png;base64,${img.base64}` : img.path;
        imgEl.alt = prompt;
        imgEl.title = `시드: ${img.seed}\n클릭하여 확대 (좌우 화살표로 탐색)`;
        imgEl.dataset.path = img.path;
        imgEl.onclick = () => showImageModalWithList(imageList, index);
        imagesDiv.appendChild(imgEl);
    });
    
    contentDiv.appendChild(imagesDiv);
    
    // 이미지가 여러 장일 때 묶음 다운로드 버튼 추가
    if (images.length > 1) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'btn btn-sm btn-secondary message-download-btn';
        downloadBtn.innerHTML = `<i class="ri-download-2-line"></i> ${images.length}장 다운로드`;
        downloadBtn.title = '이미지를 ZIP 파일로 묶어서 다운로드';
        downloadBtn.onclick = (e) => {
            e.stopPropagation();
            downloadImagesAsZip(images, prompt);
        };
        
        actionsDiv.appendChild(downloadBtn);
        contentDiv.appendChild(actionsDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============= 대화 내용 관리 =============
function getConversation() {
    const messages = [];
    const messageElements = chatMessages.querySelectorAll('.message');
    
    messageElements.forEach(msgEl => {
        const type = msgEl.classList.contains('user') ? 'user' :
                     msgEl.classList.contains('assistant') ? 'assistant' : 'system';
        
        const contentEl = msgEl.querySelector('.message-content');
        if (!contentEl) return;
        
        const textEl = contentEl.querySelector('p');
        const text = textEl ? textEl.innerHTML : '';
        
        const imagesEl = contentEl.querySelector('.message-images');
        let images = null;
        if (imagesEl) {
            images = [];
            imagesEl.querySelectorAll('img').forEach(img => {
                images.push({
                    path: img.dataset.path || img.src,
                    alt: img.alt
                });
            });
        }
        
        messages.push({ type, text, images });
    });
    
    return messages;
}

function restoreConversation(conversation) {
    const existingMessages = chatMessages.querySelectorAll('.message');
    existingMessages.forEach(msg => msg.remove());
    
    conversation.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (msg.text) {
            const p = document.createElement('p');
            p.innerHTML = msg.text;
            contentDiv.appendChild(p);
        }
        
        if (msg.images && msg.images.length > 0) {
            const imagesDiv = document.createElement('div');
            imagesDiv.className = 'message-images';
            
            // 이미지 목록 생성 (네비게이션용)
            const imageList = msg.images.map(imgData => ({
                path: imgData.path,
                metadata: { prompt: imgData.alt, seed: imgData.seed }
            }));
            
            msg.images.forEach((imgData, index) => {
                const imgEl = document.createElement('img');
                imgEl.src = imgData.path;
                imgEl.alt = imgData.alt || '';
                imgEl.dataset.path = imgData.path;
                imgEl.onclick = () => showImageModalWithList(imageList, index);
                imagesDiv.appendChild(imgEl);
            });
            
            contentDiv.appendChild(imagesDiv);
        }
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
    });
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function saveConversationToHistory(historyId) {
    if (!historyId) return;
    
    try {
        const conversation = getConversation();
        await apiCall(`/history/${historyId}/conversation`, 'PATCH', { conversation });
        console.log('대화 내용이 히스토리에 저장되었습니다.');
    } catch (error) {
        console.error('대화 내용 저장 실패:', error);
    }
}

// ============= API 호출 =============
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include'  // 쿠키 포함
    };
    
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    const response = await fetch(`/api${endpoint}`, options);
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '요청 실패');
    }
    
    return response.json();
}

// ============= LLM 버튼 비활성화/활성화 =============
const LLM_TIMEOUT = 5000;  // 5초 타임아웃

function setLlmButtonsDisabled(disabled) {
    const buttons = [
        document.getElementById('btnTemplate'),
        document.getElementById('btnTranslate'),
        document.getElementById('btnEnhance'),
        document.getElementById('btnTranslateKorean')
    ];
    
    buttons.forEach(btn => {
        if (btn) {
            btn.disabled = disabled;
            btn.style.opacity = disabled ? '0.5' : '1';
            btn.style.pointerEvents = disabled ? 'none' : 'auto';
        }
    });
    
    isLlmProcessing = disabled;
}

async function apiCallWithTimeout(endpoint, method, body, timeout = LLM_TIMEOUT) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        signal: controller.signal
    };
    
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    try {
        const response = await fetch(`/api${endpoint}`, options);
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '요청 실패');
        }
        
        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('요청 시간 초과 (5초)');
        }
        throw error;
    }
}

// ============= 이미지 생성 =============
async function generateImage(preview = false) {
    if (isGenerating) {
        addMessage('system', '⚠️ 이미 생성 요청이 진행 중입니다.');
        return;
    }
    
    const koreanText = document.getElementById('koreanInput')?.value?.trim() || '';
    let prompt = promptInput.value.trim();
    
    if (koreanText && !prompt) {
        addMessage('system', '🌐 번역 후 생성합니다...');
        const translated = await translateKoreanInput();
        if (!translated) {
            addMessage('system', '❌ 번역 실패로 생성을 중단합니다.');
            return;
        }
        prompt = promptInput.value.trim();
    }
    
    if (!prompt) {
        alert('프롬프트를 입력해주세요.');
        return;
    }
    
    isGenerating = true;
    setGenerateButtonState(true);
    
    if (koreanText && koreanText !== prompt) {
        addMessage('user', `🇰🇷 ${koreanText}\n🇺🇸 ${prompt}`);
    } else {
        addMessage('user', prompt);
    }
    
    let width, height;
    const resolutionValue = document.getElementById('resolutionSelect').value;
    
    if (resolutionValue === 'custom') {
        width = parseInt(document.getElementById('customWidth').value) || 512;
        height = parseInt(document.getElementById('customHeight').value) || 512;
    } else {
        [width, height] = resolutionValue.split('x').map(Number);
    }
    
    const requestBody = {
        prompt,
        korean_prompt: koreanText,
        width,
        height,
        steps: parseInt(document.getElementById('stepsInput').value) || 8,
        seed: parseInt(document.getElementById('seedInput').value) || -1,
        num_images: preview ? 1 : parseInt(document.getElementById('numImagesInput').value) || 1,
        auto_translate: false
    };
    
    try {
        const endpoint = preview ? '/preview' : '/generate';
        const result = await apiCall(endpoint, 'POST', requestBody);
        
        if (result.queued) {
            // 큐에 추가됨 - WebSocket으로 결과를 받음
            console.log('요청이 큐에 추가됨:', result.item_id, '순서:', result.position);
        }
    } catch (error) {
        addMessage('system', `❌ 오류: ${error.message}`, 'error');
        isGenerating = false;
        setGenerateButtonState(false);
        hideQueueStatus();
    }
}

function setGenerateButtonState(generating) {
    const btnGenerate = document.getElementById('btnGenerate');
    const btnPreview = document.getElementById('btnPreview');
    
    btnGenerate.disabled = generating;
    btnPreview.disabled = generating;
    
    if (generating) {
        btnGenerate.innerHTML = '<i class="ri-loader-4-line"></i> 생성 중...';
    } else {
        btnGenerate.innerHTML = '<i class="ri-brush-line"></i> 생성';
    }
}

// ============= 모델 관리 =============
async function loadModel(fromChat = false) {
    if (isModelLoading) {
        addMessage('system', '⚠️ 이미 모델 로딩 중입니다.');
        return;
    }
    
    // 양자화/CPU 오프로딩 설정은 설정 탭에서만 관리
    const quantization = document.getElementById('quantizationSelect')?.value || "BF16 (기본, 최고품질)";
    const modelPath = document.getElementById('modelPathInput')?.value || '';
    
    const cpuOffload = document.getElementById('cpuOffloadCheck')?.checked || false;
    
    try {
        setModelLoadingState(true);
        const offloadMsg = cpuOffload ? ' (CPU 오프로딩 사용)' : '';
        addMessage('system', `🔄 모델 로딩을 시작합니다...${offloadMsg}`);
        showProgress('모델 로딩 준비 중...', 5);

        const targetDevice = isAdmin ? (adminGpuSettings.generation_gpu || 'auto') : 'auto';
        await apiCall('/model/load', 'POST', {
            quantization,
            model_path: modelPath,
            cpu_offload: cpuOffload,
            target_device: targetDevice
        });
        
        updateModelStatus();
        updateProgress(100, '모델 로드 완료!');
        setTimeout(hideProgress, 1500);
        
        updateModelDownloadStatus();
    } catch (error) {
        addMessage('system', `❌ 모델 로드 실패: ${error.message}`, 'error');
        hideProgress();
    } finally {
        setModelLoadingState(false);
    }
}

async function unloadModel() {
    if (isModelLoading) {
        addMessage('system', '⚠️ 모델 로딩 중에는 언로드할 수 없습니다.');
        return;
    }
    
    try {
        setModelLoadingState(true);
        showProgress('모델 언로드 중...', 50);
        addMessage('system', '🔄 모델 언로드 중...');
        
        await apiCall('/model/unload', 'POST');
        updateModelStatus();
        
        updateProgress(100, '모델 언로드 완료!');
        setTimeout(hideProgress, 1000);
    } catch (error) {
        addMessage('system', `❌ 모델 언로드 실패: ${error.message}`, 'error');
        hideProgress();
    } finally {
        setModelLoadingState(false);
    }
}

function setModelLoadingState(loading) {
    isModelLoading = loading;
    
    const loadButtons = [
        document.getElementById('btnLoadModel'),
        document.getElementById('btnChatLoadModel')
    ];
    const unloadButtons = [
        document.getElementById('btnUnloadModel'),
        document.getElementById('btnChatUnloadModel')
    ];
    
    loadButtons.forEach(btn => {
        if (btn) {
            btn.disabled = loading;
            if (loading) {
                btn.innerHTML = '<i class="ri-loader-4-line"></i> 로딩...';
            } else {
                btn.innerHTML = '<i class="ri-download-line"></i> 로드';
            }
        }
    });
    
    unloadButtons.forEach(btn => {
        if (btn) btn.disabled = loading;
    });
    
    const statusBadge = document.getElementById('modelStatusBadge');
    if (statusBadge) {
        const dot = statusBadge.querySelector('.status-dot');
        const text = statusBadge.querySelector('.status-text');
        
        if (loading) {
            dot.classList.remove('online', 'offline');
            dot.classList.add('loading');
            text.textContent = '로딩 중...';
        }
    }
}

async function updateModelStatus() {
    try {
        const status = await apiCall('/status');
        
        const indicator = modelStatus.querySelector('.status-indicator');
        const text = modelStatus.querySelector('span');
        
        const statusBadge = document.getElementById('modelStatusBadge');
        const dot = statusBadge?.querySelector('.status-dot');
        const badgeText = statusBadge?.querySelector('.status-text');
        
        if (status.model_loaded) {
            indicator.classList.add('online');
            indicator.classList.remove('offline');
            text.textContent = '모델 로드됨';
            
            if (dot) {
                dot.classList.remove('offline', 'loading');
                dot.classList.add('online');
            }
            if (badgeText) {
                if (status.current_model) {
                    badgeText.textContent = `✓ ${status.current_model}`;
                    statusBadge.title = status.current_model;
                } else {
                    badgeText.textContent = '모델 로드됨';
                    statusBadge.title = '';
                }
            }
        } else {
            indicator.classList.remove('online');
            indicator.classList.add('offline');
            text.textContent = '모델 미로드';
            
            if (dot) {
                dot.classList.remove('online', 'loading');
                dot.classList.add('offline');
            }
            if (badgeText) badgeText.textContent = '모델 미로드';
            if (statusBadge) statusBadge.title = '';
        }
        
        // 관리자 상태 업데이트
        if (status.is_admin !== undefined) {
            isAdmin = status.is_admin;
            updateAdminUI();
        }
        
        // 접속자 수 업데이트
        if (status.connected_users) {
            updateUserCount(status.connected_users);
        }
    } catch (error) {
        console.error('상태 업데이트 실패:', error);
    }
}

// ============= 관리자 UI 업데이트 =============
function updateAdminUI() {
    const adminNotice = document.getElementById('adminNotice');
    const llmSettingsSection = document.getElementById('llmSettingsSection');
    const sessionManagementSection = document.getElementById('sessionManagementSection');
    const systemPromptsSection = document.getElementById('systemPromptsSection');
    const autoUnloadSection = document.getElementById('autoUnloadSection');
    const editAutoUnloadSection = document.getElementById('editAutoUnloadSection');
    const gpuManagementSection = document.getElementById('gpuManagementSection');
    const quantizationSelect = document.getElementById('quantizationSelect');
    const cpuOffloadCheck = document.getElementById('cpuOffloadCheck');
    const editQuantizationSelectSettings = document.getElementById('editQuantizationSelectSettings');
    const editCpuOffloadCheckSettings = document.getElementById('editCpuOffloadCheckSettings');
    
    // 시스템 프롬프트는 개인화되므로 항상 활성화
    if (systemPromptsSection) {
        systemPromptsSection.querySelectorAll('input, select, textarea, button').forEach(el => {
            el.disabled = false;
            el.style.display = '';
        });
    }
    
    if (isAdmin) {
        // 관리자: LLM 설정 및 자동 언로드 설정 변경 가능
        if (adminNotice) adminNotice.style.display = 'none';
        if (llmSettingsSection) {
            llmSettingsSection.querySelectorAll('input, select, button').forEach(el => {
                el.disabled = false;
                el.style.display = '';
            });
        }
        if (autoUnloadSection) {
            autoUnloadSection.querySelectorAll('input, button').forEach(el => {
                el.disabled = false;
                el.style.display = '';
            });
        }
        if (editAutoUnloadSection) {
            editAutoUnloadSection.querySelectorAll('input, button').forEach(el => {
                el.disabled = false;
                el.style.display = '';
            });
        }
        if (sessionManagementSection) {
            sessionManagementSection.style.display = 'block';
            loadSessionList();
        }

        if (gpuManagementSection) {
            gpuManagementSection.style.display = 'block';
        }

        // 양자화/CPU 오프로딩은 관리자만 변경 가능
        [quantizationSelect, cpuOffloadCheck, editQuantizationSelectSettings, editCpuOffloadCheckSettings].forEach(el => {
            if (el) el.disabled = false;
        });
    } else {
        // 일반 사용자: LLM 설정 및 자동 언로드 설정 읽기 전용
        if (adminNotice) adminNotice.style.display = 'block';
        if (llmSettingsSection) {
            llmSettingsSection.querySelectorAll('input, select').forEach(el => {
                el.disabled = true;
            });
            llmSettingsSection.querySelectorAll('button').forEach(el => {
                el.style.display = 'none';
            });
        }
        if (autoUnloadSection) {
            autoUnloadSection.querySelectorAll('input').forEach(el => {
                el.disabled = true;
            });
            autoUnloadSection.querySelectorAll('button').forEach(el => {
                el.style.display = 'none';
            });
        }
        if (editAutoUnloadSection) {
            editAutoUnloadSection.querySelectorAll('input').forEach(el => {
                el.disabled = true;
            });
            editAutoUnloadSection.querySelectorAll('button').forEach(el => {
                el.style.display = 'none';
            });
        }
        if (sessionManagementSection) {
            sessionManagementSection.style.display = 'none';
        }

        if (gpuManagementSection) {
            gpuManagementSection.style.display = 'none';
        }

        // 양자화/CPU 오프로딩은 관리자만 변경 가능
        [quantizationSelect, cpuOffloadCheck, editQuantizationSelectSettings, editCpuOffloadCheckSettings].forEach(el => {
            if (el) el.disabled = true;
        });
    }
}

// ============= GPU 관리 (관리자 전용) =============
function setSelectOptions(selectEl, options, selectedValue) {
    if (!selectEl) return;
    selectEl.innerHTML = '';
    options.forEach(optVal => {
        const opt = document.createElement('option');
        opt.value = optVal;
        opt.textContent = optVal;
        selectEl.appendChild(opt);
    });
    if (selectedValue && options.includes(selectedValue)) {
        selectEl.value = selectedValue;
    }
}

function renderGpuStatus(data) {
    const summaryEl = document.getElementById('gpuStatusSummary');
    const listEl = document.getElementById('gpuStatusList');
    if (!summaryEl || !listEl) return;

    const gpus = data?.gpus || [];
    const models = data?.models || {};
    const currentSettings = data?.current_settings || {};

    const genDevice = models?.generation?.device || 'N/A';
    const editDevice = models?.edit?.device || 'N/A';
    const editQuant = models?.edit?.quantization ? String(models.edit.quantization).toUpperCase() : 'BF16';

    summaryEl.textContent =
        `생성 모델: ${models?.generation?.loaded ? '로드됨' : '미로드'} (${genDevice}) / ` +
        `편집 모델: ${models?.edit?.loaded ? '로드됨' : '미로드'} (${editDevice}, ${editQuant}) / ` +
        `설정: 생성=${currentSettings.generation_gpu || 'auto'}, 편집=${currentSettings.edit_gpu || 'auto'}`;

    listEl.innerHTML = '';
    if (!gpus.length) {
        const empty = document.createElement('div');
        empty.className = 'gpu-status-item';
        empty.innerHTML = `<div class="gpu-status-item-title"><span>GPU 정보 없음</span><span></span></div>
<div class="gpu-status-item-sub">CUDA 사용 불가이거나 GPU가 감지되지 않았습니다.</div>`;
        listEl.appendChild(empty);
        return;
    }

    gpus.forEach(gpu => {
        const mem = gpu.memory || {};
        const util = gpu.utilization || {};
        const loadedModels = gpu.loaded_models || [];
        const item = document.createElement('div');
        item.className = 'gpu-status-item';
        item.innerHTML = `
            <div class="gpu-status-item-title">
                <span>GPU ${gpu.id}: ${gpu.name || ''}</span>
                <span>${(mem.used_gb ?? mem.allocated_gb ?? 0).toFixed(2)}GB / ${(mem.total_gb ?? 0).toFixed(2)}GB</span>
            </div>
            <div class="gpu-status-item-sub">
                사용(used): ${(mem.used_gb ?? 0).toFixed(2)}GB / 예약(reserved): ${(mem.reserved_gb ?? 0).toFixed(2)}GB / 사용률: ${(mem.usage_percent ?? 0).toFixed(1)}%<br/>
                사용률(GPU): ${util.gpu_percent ?? 'N/A'}% / 사용률(VRAM): ${util.memory_percent ?? 'N/A'}% (${util.source || 'unknown'})<br/>
                로드된 모델: ${loadedModels.length ? loadedModels.join(', ') : '없음'}
            </div>
        `;
        listEl.appendChild(item);
    });
}

async function loadAdminGpuPanel() {
    if (!isAdmin) return;
    try {
        const data = await apiCall('/admin/gpu-status');

        // 서버가 내려주는 디바이스 목록 사용
        adminAvailableDevices = data?.available_devices || adminAvailableDevices;
        if (!Array.isArray(adminAvailableDevices) || adminAvailableDevices.length === 0) {
            const dev = await apiCall('/admin/available-devices');
            adminAvailableDevices = dev?.devices || ['auto', 'cpu'];
        }

        // 현재 설정 반영
        if (data?.current_settings) {
            adminGpuSettings.generation_gpu = data.current_settings.generation_gpu || adminGpuSettings.generation_gpu;
            adminGpuSettings.edit_gpu = data.current_settings.edit_gpu || adminGpuSettings.edit_gpu;
        }

        const genSelect = document.getElementById('generationGpuSelect');
        const editSelect = document.getElementById('editGpuSelect');
        setSelectOptions(genSelect, adminAvailableDevices, adminGpuSettings.generation_gpu);
        setSelectOptions(editSelect, adminAvailableDevices, adminGpuSettings.edit_gpu);

        renderGpuStatus(data);
    } catch (error) {
        console.error('GPU 상태 로드 실패:', error);
        const summaryEl = document.getElementById('gpuStatusSummary');
        if (summaryEl) summaryEl.textContent = `GPU 상태 로드 실패: ${error.message}`;
    }
}

async function saveAdminGpuSettings() {
    if (!isAdmin) {
        addMessage('system', '❌ GPU 설정은 관리자만 변경할 수 있습니다.', 'error');
        return;
    }
    const genSelect = document.getElementById('generationGpuSelect');
    const editSelect = document.getElementById('editGpuSelect');
    const generation_gpu = genSelect?.value || 'auto';
    const edit_gpu = editSelect?.value || 'auto';

    try {
        const result = await apiCall('/admin/gpu-settings', 'POST', { generation_gpu, edit_gpu });
        adminGpuSettings.generation_gpu = result?.settings?.generation_gpu || generation_gpu;
        adminGpuSettings.edit_gpu = result?.settings?.edit_gpu || edit_gpu;
        addMessage('system', `✅ GPU 설정 저장됨 (생성=${adminGpuSettings.generation_gpu}, 편집=${adminGpuSettings.edit_gpu})`);
        await loadAdminGpuPanel();
    } catch (error) {
        addMessage('system', `❌ GPU 설정 저장 실패: ${error.message}`, 'error');
    }
}

// ============= 세션 관리 (관리자 전용) =============
async function loadSessionList() {
    try {
        const result = await apiCall('/admin/sessions');
        const sessionList = document.getElementById('sessionList');
        if (!sessionList) return;
        
        // 헤더 유지하고 나머지 삭제
        const header = sessionList.querySelector('.session-list-header');
        sessionList.innerHTML = '';
        if (header) sessionList.appendChild(header);
        
        const rows = result.users || [];
        rows.forEach(user => {
            const item = document.createElement('div');
            item.className = 'session-list-item';
            const usernameDisplay = user.username || (user.user_id ? `user_${user.user_id}` : '알 수 없음');
            const idDisplay = user.data_id || '';
            item.innerHTML = `
                <span class="session-id" title="${idDisplay}">${idDisplay}</span>
                <span class="session-user">${usernameDisplay}</span>
                <span class="session-activity">${formatDate(user.last_activity)}</span>
                <span class="session-size">${user.data_size || ''}</span>
                <button class="btn btn-xs btn-danger" onclick="deleteSession('${idDisplay}')">
                    <i class="ri-delete-bin-line"></i>
                </button>
            `;
            sessionList.appendChild(item);
        });
    } catch (error) {
        console.error('세션 목록 로드 실패:', error);
    }
}

async function deleteSession(sessionId) {
    if (!confirm('이 사용자의 현재 접속(WebSocket)과 대기열 요청을 제거하시겠습니까?')) return;
    
    try {
        await apiCall(`/admin/sessions/${sessionId}`, 'DELETE');
        loadSessionList();
        addMessage('system', '✅ 사용자 접속이 정리되었습니다.');
    } catch (error) {
        addMessage('system', `❌ 사용자 정리 실패: ${error.message}`, 'error');
    }
}

// ============= 프롬프트 도구 =============
function isKorean(text) {
    const koreanRegex = /[가-힣]/;
    return koreanRegex.test(text);
}

async function translateKoreanInput() {
    const koreanInputEl = document.getElementById('koreanInput');
    const koreanText = koreanInputEl?.value?.trim();
    const statusEl = document.getElementById('translateStatus');
    
    if (!koreanText) {
        addMessage('system', '⚠️ 한국어 입력창에 텍스트를 입력해주세요.');
        return false;
    }
    
    if (isTranslating || isLlmProcessing) {
        return false;
    }
    
    if (!isKorean(koreanText)) {
        document.getElementById('promptInput').value = koreanText;
        if (statusEl) {
            statusEl.textContent = '✓ 복사됨';
            statusEl.className = 'translate-status success';
            setTimeout(() => {
                statusEl.textContent = '';
                statusEl.className = 'translate-status';
            }, 2000);
        }
        return true;
    }
    
    try {
        isTranslating = true;
        setLlmButtonsDisabled(true);
        if (statusEl) {
            statusEl.textContent = '번역 중...';
            statusEl.className = 'translate-status translating';
        }
        
        const result = await apiCallWithTimeout('/translate', 'POST', { text: koreanText });
        
        if (result.success) {
            document.getElementById('promptInput').value = result.translated;
            if (statusEl) {
                statusEl.textContent = '✓ 번역됨';
                statusEl.className = 'translate-status success';
                setTimeout(() => {
                    if (statusEl.textContent === '✓ 번역됨') {
                        statusEl.textContent = '';
                        statusEl.className = 'translate-status';
                    }
                }, 2000);
            }
            return true;
        }
        return false;
    } catch (error) {
        if (statusEl) {
            statusEl.textContent = '번역 실패';
            statusEl.className = 'translate-status error';
        }
        addMessage('system', `❌ 번역 실패: ${error.message}`, 'error');
        return false;
    } finally {
        isTranslating = false;
        setLlmButtonsDisabled(false);
    }
}

async function translatePrompt() {
    const koreanInputEl = document.getElementById('koreanInput');
    const text = koreanInputEl?.value?.trim() || promptInput.value.trim();
    if (!text) return;
    
    if (isLlmProcessing) return;
    
    try {
        setLlmButtonsDisabled(true);
        addMessage('system', '🌐 번역 중...');
        const result = await apiCallWithTimeout('/translate', 'POST', { text });
        
        if (result.success) {
            promptInput.value = result.translated;
            addMessage('system', '✅ 번역 완료');
        }
    } catch (error) {
        addMessage('system', `❌ 번역 실패: ${error.message}`, 'error');
    } finally {
        setLlmButtonsDisabled(false);
    }
}

async function enhancePrompt() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;
    
    if (isLlmProcessing) return;
    
    const koreanInputEl = document.getElementById('koreanInput');
    const statusEl = document.getElementById('translateStatus');
    
    try {
        setLlmButtonsDisabled(true);
        addMessage('system', '✨ 프롬프트 향상 중...');
        const result = await apiCallWithTimeout('/enhance', 'POST', { prompt, style: '기본' });
        
        if (result.success) {
            promptInput.value = result.enhanced;
            addMessage('system', '✅ 프롬프트 향상 완료');
            
            if (koreanInputEl) {
                try {
                    if (statusEl) {
                        statusEl.textContent = '한국어 변환 중...';
                        statusEl.className = 'translate-status translating';
                    }
                    
                    const reverseResult = await apiCallWithTimeout('/translate-reverse', 'POST', { text: result.enhanced });
                    
                    if (reverseResult.success) {
                        koreanInputEl.value = reverseResult.translated;
                        addMessage('system', '🔄 한국어 프롬프트도 업데이트됨');
                        
                        if (statusEl) {
                            statusEl.textContent = '✓ 동기화됨';
                            statusEl.className = 'translate-status success';
                            setTimeout(() => {
                                if (statusEl.textContent === '✓ 동기화됨') {
                                    statusEl.textContent = '';
                                    statusEl.className = 'translate-status';
                                }
                            }, 2000);
                        }
                    }
                } catch (reverseError) {
                    console.error('역번역 실패:', reverseError);
                    if (statusEl) {
                        statusEl.textContent = '';
                        statusEl.className = 'translate-status';
                    }
                }
            }
        }
    } catch (error) {
        addMessage('system', `❌ 향상 실패: ${error.message}`, 'error');
    } finally {
        setLlmButtonsDisabled(false);
    }
}

async function loadTemplates() {
    try {
        const result = await apiCall('/templates');
        templates = result.templates;
        
        const list = document.getElementById('templateList');
        list.innerHTML = '';
        
        for (const [name, template] of Object.entries(templates)) {
            const item = document.createElement('div');
            item.className = 'template-item';
            item.innerHTML = `
                <div class="template-item-name">${name}</div>
                <div class="template-item-preview">${template.prompt.substring(0, 80)}...</div>
            `;
            item.onclick = () => applyTemplate(name, template);
            list.appendChild(item);
        }
    } catch (error) {
        console.error('템플릿 로드 실패:', error);
    }
}

function applyTemplate(name, template) {
    let prompt = template.prompt;
    
    if (template.variables) {
        for (const [key, value] of Object.entries(template.variables)) {
            prompt = prompt.replace(`{${key}}`, value);
        }
    }
    
    promptInput.value = prompt;
    const koreanInputEl = document.getElementById('koreanInput');
    if (koreanInputEl) koreanInputEl.value = '';
    
    closeModal('templateModal');
    addMessage('system', `✅ 템플릿 적용: ${name} (영어 프롬프트 직접 사용)`);
}

// ============= 양자화 옵션 로드 =============
async function loadQuantizationOptions() {
    try {
        const result = await apiCall('/settings');
        const settingsSelect = document.getElementById('quantizationSelect');
        
        if (result.quantization_options) {
            if (settingsSelect) {
                settingsSelect.innerHTML = '';
                result.quantization_options.forEach(option => {
                    const opt = document.createElement('option');
                    opt.value = option;
                    opt.textContent = option;
                    settingsSelect.appendChild(opt);
                });
            }
            
            console.log('양자화 옵션 로드 완료:', result.quantization_options.length + '개');
            
            updateModelDownloadStatus();
        }

        // 서버에 저장된 모델 설정값 반영
        const savedQuant = result.quantization;
        const savedCpuOffload = result.cpu_offload;
        const savedEditQuant = result.edit_quantization;
        const savedEditCpuOffload = result.edit_cpu_offload;

        if (settingsSelect && savedQuant && Array.from(settingsSelect.options).some(o => o.value === savedQuant)) {
            settingsSelect.value = savedQuant;
        }

        const cpuOffloadCheck = document.getElementById('cpuOffloadCheck');
        if (cpuOffloadCheck && typeof savedCpuOffload === 'boolean') {
            cpuOffloadCheck.checked = savedCpuOffload;
        }

        const editCpuOffloadCheckSettings = document.getElementById('editCpuOffloadCheckSettings');
        if (editCpuOffloadCheckSettings && typeof savedEditCpuOffload === 'boolean') {
            editCpuOffloadCheckSettings.checked = savedEditCpuOffload;
        }

        if (savedEditQuant) {
            pendingEditQuantizationValue = savedEditQuant;
            const editSelect = document.getElementById('editQuantizationSelectSettings');
            if (editSelect && Array.from(editSelect.options).some(o => o.value === savedEditQuant)) {
                editSelect.value = savedEditQuant;
                pendingEditQuantizationValue = null;
            }
        }
        
        // 관리자 상태 업데이트
        if (result.is_admin !== undefined) {
            isAdmin = result.is_admin;
            updateAdminUI();
        }
    } catch (error) {
        console.error('양자화 옵션 로드 실패:', error);
    }
}

// ============= 모델 설정 저장 (관리자 전용) =============
async function saveModelSettings() {
    if (!isAdmin) return;

    const quantization = document.getElementById('quantizationSelect')?.value || "BF16 (기본, 최고품질)";
    const cpuOffload = document.getElementById('cpuOffloadCheck')?.checked || false;
    const editQuantization = document.getElementById('editQuantizationSelectSettings')?.value || "BF16 (기본, 최고품질)";
    const editCpuOffload = document.getElementById('editCpuOffloadCheckSettings')?.checked ?? true;

    try {
        await apiCall('/settings', 'POST', {
            quantization,
            cpu_offload: cpuOffload,
            edit_quantization: editQuantization,
            edit_cpu_offload: editCpuOffload
        });
    } catch (error) {
        console.error('모델 설정 저장 실패:', error);
    }
}

async function updateModelDownloadStatus() {
    try {
        const result = await apiCall('/model-status');
        const status = result.status || {};
        
        const settingsSelect = document.getElementById('quantizationSelect');
        
        if (settingsSelect) {
            Array.from(settingsSelect.options).forEach(opt => {
                const optionName = opt.value;
                const isDownloaded = status[optionName] || false;

                let text = opt.textContent.replace(/^[✓⬇]\s*/, '');

                if (isDownloaded) {
                    opt.textContent = `✓ ${text}`;
                    opt.style.color = '#22c55e';
                } else {
                    opt.textContent = `⬇ ${text}`;
                    opt.style.color = '';
                }

                const statusText = isDownloaded ? '(다운로드됨)' : '(미다운로드)';
                opt.title = `${optionName} ${statusText}`;
            });
        }
        
        console.log('모델 다운로드 상태 업데이트 완료');
    } catch (error) {
        console.error('모델 다운로드 상태 확인 실패:', error);
    }
}

// ============= 갤러리 =============
// 갤러리 데이터 저장 (다운로드용)
let galleryData = {
    images: [],
    groups: []
};

async function loadGallery() {
    try {
        const result = await apiCall('/gallery');
        const grid = document.getElementById('galleryGrid');
        grid.innerHTML = '';
        
        // 갤러리 데이터 저장
        galleryData.images = result.images;
        
        // 갤러리 이미지 목록 생성 (네비게이션용)
        const galleryImageList = result.images.map(img => ({
            path: img.path,
            metadata: img.metadata
        }));
        
        result.images.forEach((img, index) => {
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
                <img src="${img.path}" alt="${img.filename}">
                <div class="gallery-item-overlay">
                    <span>${img.filename}</span>
                </div>
            `;
            item.onclick = () => showImageModalWithList(galleryImageList, index);
            grid.appendChild(item);
        });
        
        // 다운로드 메뉴 업데이트
        updateGalleryDownloadMenu(result.images);
        
    } catch (error) {
        console.error('갤러리 로드 실패:', error);
    }
}

// 갤러리 다운로드 메뉴 업데이트
function updateGalleryDownloadMenu(images) {
    const countEl = document.getElementById('downloadAllCount');
    const groupListEl = document.getElementById('galleryGroupList');
    
    // 전체 개수 표시
    if (countEl) {
        countEl.textContent = `${images.length}장`;
    }
    
    // 생성 프로세스별 그룹핑
    const groups = groupImagesByProcess(images);
    galleryData.groups = groups;
    
    // 그룹 목록 렌더링
    if (groupListEl) {
        if (groups.length === 0) {
            groupListEl.innerHTML = '<div class="dropdown-empty">이미지가 없습니다</div>';
        } else {
            groupListEl.innerHTML = groups.map((group, index) => `
                <button class="dropdown-group-item" onclick="downloadGalleryGroup(${index})" data-prompt="${escapeHtml(group.prompt)}">
                    <div class="group-thumbnail">
                        <img src="${group.thumbnail}" alt="미리보기">
                    </div>
                    <div class="group-info">
                        <div class="group-prompt">${escapeHtml(group.promptShort)}</div>
                    </div>
                    <span class="group-count">${group.images.length}장</span>
                </button>
            `).join('');
            
            // 툴팁 이벤트 추가
            groupListEl.querySelectorAll('.dropdown-group-item').forEach(item => {
                item.addEventListener('mouseenter', showGroupTooltip);
                item.addEventListener('mouseleave', hideGroupTooltip);
                item.addEventListener('mousemove', moveGroupTooltip);
            });
        }
    }
}

// 그룹 툴팁 표시
function showGroupTooltip(e) {
    const prompt = e.currentTarget.dataset.prompt;
    if (!prompt) return;
    
    let tooltip = document.getElementById('groupTooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'groupTooltip';
        tooltip.className = 'floating-tooltip';
        document.body.appendChild(tooltip);
    }
    
    tooltip.textContent = prompt;
    tooltip.style.display = 'block';
    
    positionTooltip(tooltip, e);
}

// 툴팁 위치 조정
function positionTooltip(tooltip, e) {
    const padding = 15;
    const tooltipWidth = 300;
    
    let x = e.clientX - tooltipWidth - padding;
    let y = e.clientY;
    
    // 왼쪽 화면 경계 체크
    if (x < padding) {
        x = e.clientX + padding;
    }
    
    // 하단 경계 체크
    const tooltipHeight = tooltip.offsetHeight || 100;
    if (y + tooltipHeight > window.innerHeight - padding) {
        y = window.innerHeight - tooltipHeight - padding;
    }
    
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}

// 툴팁 이동
function moveGroupTooltip(e) {
    const tooltip = document.getElementById('groupTooltip');
    if (tooltip && tooltip.style.display === 'block') {
        positionTooltip(tooltip, e);
    }
}

// 툴팁 숨김
function hideGroupTooltip() {
    const tooltip = document.getElementById('groupTooltip');
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

// 이미지를 프롬프트별로 그룹핑
function groupImagesByProcess(images) {
    const groups = [];
    const groupMap = new Map();
    
    images.forEach(img => {
        const metadata = img.metadata || {};
        const prompt = metadata.prompt || 'unknown';
        
        // 프롬프트를 그룹 키로 사용 (공백 정규화)
        const groupKey = prompt.trim().toLowerCase();
        
        if (!groupMap.has(groupKey)) {
            groupMap.set(groupKey, {
                prompt: prompt,
                promptShort: prompt.length > 35 ? prompt.substring(0, 35) + '...' : prompt,
                thumbnail: img.path, // 첫 번째 이미지를 썸네일로
                images: []
            });
        }
        
        groupMap.get(groupKey).images.push(img);
    });
    
    // Map을 배열로 변환하고 이미지 개수순 정렬 (많은 순)
    groupMap.forEach((group, key) => {
        groups.push(group);
    });
    
    groups.sort((a, b) => b.images.length - a.images.length);
    
    return groups;
}

// 시간 키 포맷팅
function formatTimeKey(timeKey) {
    if (!timeKey || timeKey === 'unknown') return '알 수 없음';
    
    // YYYYMMDD_HHMMSS 형식
    const match = timeKey.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})?/);
    if (match) {
        const [, year, month, day, hour, minute] = match;
        return `${month}/${day} ${hour}:${minute}`;
    }
    return timeKey;
}

// 갤러리 전체 다운로드
async function downloadAllGalleryImages() {
    if (galleryData.images.length === 0) {
        alert('다운로드할 이미지가 없습니다.');
        return;
    }
    
    closeGalleryDropdown();
    
    const images = galleryData.images.map(img => ({
        path: img.path,
        seed: img.metadata?.seed || 0
    }));
    
    await downloadImagesAsZipWithStatus(images, 'gallery_all', '갤러리 전체');
}

// 갤러리 그룹별 다운로드
async function downloadGalleryGroup(groupIndex) {
    const group = galleryData.groups[groupIndex];
    if (!group || group.images.length === 0) {
        alert('다운로드할 이미지가 없습니다.');
        return;
    }
    
    closeGalleryDropdown();
    
    const images = group.images.map(img => ({
        path: img.path,
        seed: img.metadata?.seed || 0
    }));
    
    const promptShort = group.prompt.substring(0, 30).replace(/[^a-zA-Z0-9가-힣]/g, '_');
    await downloadImagesAsZipWithStatus(images, `gallery_${promptShort}`, group.promptShort);
}

// 상태 표시와 함께 ZIP 다운로드 (갤러리용)
async function downloadImagesAsZipWithStatus(images, prefix, description) {
    if (typeof JSZip === 'undefined') {
        alert('ZIP 라이브러리를 불러오지 못했습니다. 페이지를 새로고침해주세요.');
        return;
    }
    
    // 진행 상태 알림
    const statusDiv = document.createElement('div');
    statusDiv.className = 'download-status-toast';
    statusDiv.innerHTML = `
        <div class="download-status-content">
            <i class="ri-download-2-line"></i>
            <span id="downloadStatusText">📦 ${images.length}장의 이미지를 다운로드 준비 중...</span>
        </div>
    `;
    document.body.appendChild(statusDiv);
    
    const statusText = document.getElementById('downloadStatusText');
    
    try {
        const zip = new JSZip();
        const folder = zip.folder('images');
        
        let successCount = 0;
        for (let i = 0; i < images.length; i++) {
            const img = images[i];
            try {
                statusText.textContent = `📦 다운로드 중... (${i + 1}/${images.length})`;
                
                const response = await fetch(img.path);
                const blob = await response.blob();
                
                const filename = img.path.split('/').pop() || `image_${i + 1}.png`;
                folder.file(filename, blob);
                successCount++;
            } catch (error) {
                console.error(`이미지 ${i + 1} 다운로드 실패:`, error);
            }
        }
        
        if (successCount === 0) {
            statusText.textContent = '❌ 이미지 다운로드에 실패했습니다.';
            setTimeout(() => statusDiv.remove(), 3000);
            return;
        }
        
        statusText.textContent = `📦 ZIP 파일 생성 중... (${successCount}장)`;
        
        const content = await zip.generateAsync({ type: 'blob' });
        
        // 파일명 생성
        const date = new Date();
        const dateStr = `${date.getFullYear()}${(date.getMonth()+1).toString().padStart(2,'0')}${date.getDate().toString().padStart(2,'0')}`;
        const timeStr = `${date.getHours().toString().padStart(2,'0')}${date.getMinutes().toString().padStart(2,'0')}`;
        const zipFilename = `${prefix}_${dateStr}_${timeStr}.zip`;
        
        // 다운로드
        const url = URL.createObjectURL(content);
        const link = document.createElement('a');
        link.href = url;
        link.download = zipFilename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        statusText.textContent = `✅ ${successCount}장 다운로드 완료!`;
        setTimeout(() => statusDiv.remove(), 2000);
        
    } catch (error) {
        console.error('ZIP 생성 실패:', error);
        statusText.textContent = `❌ ZIP 생성 실패: ${error.message}`;
        setTimeout(() => statusDiv.remove(), 3000);
    }
}

// 갤러리 드롭다운 토글
function toggleGalleryDropdown() {
    const dropdown = document.getElementById('galleryDownloadDropdown');
    dropdown.classList.toggle('open');
}

function closeGalleryDropdown() {
    const dropdown = document.getElementById('galleryDownloadDropdown');
    dropdown.classList.remove('open');
}

// ============= 갤러리에서 이미지 선택 (편집용) =============
async function openGallerySelectModal() {
    const modal = document.getElementById('gallerySelectModal');
    const grid = document.getElementById('gallerySelectGrid');
    
    if (!modal || !grid) return;
    
    // 갤러리 데이터 로드
    try {
        const result = await apiCall('/gallery');
        
        if (result.images.length === 0) {
            grid.innerHTML = '<div class="gallery-select-empty"><i class="ri-image-line"></i><p>갤러리에 이미지가 없습니다.</p></div>';
        } else {
            grid.innerHTML = result.images.map(img => `
                <div class="gallery-select-item" data-path="${img.path}">
                    <img src="${img.path}" alt="${img.filename}">
                    <div class="select-overlay">
                        <span>${img.filename}</span>
                    </div>
                </div>
            `).join('');
            
            // 클릭 이벤트 추가
            grid.querySelectorAll('.gallery-select-item').forEach(item => {
                item.addEventListener('click', () => {
                    selectImageFromGallery(item.dataset.path);
                });
            });
        }
        
        modal.classList.add('active');
        
    } catch (error) {
        console.error('갤러리 로드 실패:', error);
        addEditMessage('system', `❌ 갤러리 로드 실패: ${error.message}`);
    }
}

// 갤러리에서 이미지 선택
function selectImageFromGallery(imagePath) {
    // 모달 닫기
    closeModal('gallerySelectModal');
    
    // 이미지를 편집 탭에 로드
    loadImageToEditTab(imagePath);
}

// 이미지를 편집 탭에 로드
async function loadImageToEditTab(imagePath) {
    try {
        // 이미지를 fetch하여 File 객체로 변환
        const response = await fetch(imagePath);
        const blob = await response.blob();
        let filename = 'image.png';
        if (typeof imagePath === 'string' && !imagePath.startsWith('data:')) {
            const last = imagePath.split('/').pop();
            if (last && last.length < 200) {
                filename = last.split('?')[0] || 'image.png';
            }
        }
        editImageFile = new File([blob], filename, { type: blob.type || 'image/png' });
        
        // 미리보기 표시
        const preview = document.getElementById('editUploadPreview');
        const placeholder = document.getElementById('editUploadPlaceholder');
        const img = document.getElementById('editPreviewImage');
        
        img.src = imagePath;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
        
        addEditMessage('system', '✅ 이미지가 로드되었습니다. 편집 지시어를 입력하세요.');

        // 바로 이어서 입력할 수 있게 포커스
        const koreanInput = document.getElementById('editKoreanInput');
        if (koreanInput) koreanInput.focus();
        
    } catch (error) {
        console.error('이미지 로드 실패:', error);
        addEditMessage('system', `❌ 이미지 로드 실패: ${error.message}`);
    }
}

function continueEditFromMessageImage(imageSrc) {
    if (!imageSrc) return;
    // 편집 결과 이미지(서버 경로/데이터URL) 모두 지원
    loadImageToEditTab(imageSrc);
}

// 이미지 뷰어에서 편집 탭으로 이동
function editCurrentImage() {
    const currentPath = imagePreviewState.currentPath;
    if (!currentPath) {
        alert('편집할 이미지가 없습니다.');
        return;
    }
    
    // 모달 닫기
    closeImageModal();
    
    // 편집 탭으로 전환
    switchTab('edit');
    
    // 이미지 로드
    loadImageToEditTab(currentPath);
}

// ============= 히스토리 =============
async function loadHistory() {
    try {
        const result = await apiCall('/history');
        const list = document.getElementById('historyList');
        list.innerHTML = '';
        
        result.history.forEach(entry => {
            const hasConversation = entry.conversation && entry.conversation.length > 0;
            const hasKorean = entry.korean_prompt && entry.korean_prompt.trim();
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div class="history-item-header">
                    <span class="history-item-time">${formatDate(entry.timestamp)}</span>
                    <div class="item-actions">
                        ${hasConversation ? `<button class="btn btn-primary" onclick="restoreHistoryConversation('${entry.id}')" title="대화 내용을 복원합니다"><i class="ri-chat-history-line"></i> 대화 복원</button>` : ''}
                        <button class="btn btn-secondary" onclick="useHistoryEntry('${entry.id}')">사용</button>
                    </div>
                </div>
                ${hasKorean ? `<div class="history-item-korean"><span class="lang-badge kr">🇰🇷</span> ${escapeHtml(entry.korean_prompt)}</div>` : ''}
                <div class="history-item-prompt"><span class="lang-badge us">🇺🇸</span> ${escapeHtml(entry.prompt)}</div>
                ${hasConversation ? `<div class="history-item-badge"><i class="ri-chat-3-line"></i> 대화 ${entry.conversation.length}개 메시지</div>` : ''}
            `;
            list.appendChild(item);
        });
    } catch (error) {
        console.error('히스토리 로드 실패:', error);
    }
}

async function useHistoryEntry(historyId) {
    try {
        const result = await apiCall(`/history/${historyId}`);
        const entry = result.history;
        
        promptInput.value = entry.prompt;
        
        const koreanInputEl = document.getElementById('koreanInput');
        if (koreanInputEl) {
            koreanInputEl.value = entry.korean_prompt || '';
        }
        
        switchTab('chat');
        
        if (entry.korean_prompt) {
            addMessage('system', '✅ 프롬프트 적용됨 (🇰🇷 한국어 + 🇺🇸 영어)');
        } else {
            addMessage('system', '✅ 프롬프트 적용됨 (🇺🇸 영어)');
        }
    } catch (error) {
        console.error('히스토리 사용 실패:', error);
        addMessage('system', `❌ 히스토리 로드 실패: ${error.message}`, 'error');
    }
}

function useHistoryPrompt(prompt) {
    promptInput.value = prompt;
    const koreanInputEl = document.getElementById('koreanInput');
    if (koreanInputEl) koreanInputEl.value = '';
    switchTab('chat');
    addMessage('system', '✅ 프롬프트 적용됨');
}

async function restoreHistoryConversation(historyId) {
    try {
        const result = await apiCall(`/history/${historyId}`);
        const entry = result.history;
        
        if (entry.conversation && entry.conversation.length > 0) {
            if (!confirm('현재 대화 내용을 지우고 히스토리의 대화를 복원하시겠습니까?')) {
                return;
            }
            
            promptInput.value = entry.prompt;
            
            const koreanInputEl = document.getElementById('koreanInput');
            if (koreanInputEl) {
                koreanInputEl.value = entry.korean_prompt || '';
            }
            
            if (entry.settings) {
                if (entry.settings.width && entry.settings.height) {
                    const resSelect = document.getElementById('resolutionSelect');
                    const resValue = `${entry.settings.width}x${entry.settings.height}`;
                    if ([...resSelect.options].some(opt => opt.value === resValue)) {
                        resSelect.value = resValue;
                        document.getElementById('customResolution').style.display = 'none';
                    } else {
                        resSelect.value = 'custom';
                        document.getElementById('customResolution').style.display = 'flex';
                        document.getElementById('customWidth').value = entry.settings.width;
                        document.getElementById('customHeight').value = entry.settings.height;
                    }
                }
                if (entry.settings.seed) {
                    document.getElementById('seedInput').value = entry.settings.seed;
                }
                if (entry.settings.steps) {
                    document.getElementById('stepsInput').value = entry.settings.steps;
                }
            }
            
            restoreConversation(entry.conversation);
            
            switchTab('chat');
            
            addMessage('system', '✅ 히스토리에서 대화가 복원되었습니다.');
        } else {
            addMessage('system', '⚠️ 이 히스토리에는 저장된 대화 내용이 없습니다.');
            switchTab('chat');
        }
    } catch (error) {
        console.error('대화 복원 실패:', error);
        addMessage('system', `❌ 대화 복원 실패: ${error.message}`, 'error');
    }
}

async function clearHistory() {
    if (!confirm('모든 히스토리를 삭제하시겠습니까?')) return;
    
    try {
        await apiCall('/history', 'DELETE');
        loadHistory();
        addMessage('system', '✅ 히스토리 삭제됨');
    } catch (error) {
        addMessage('system', `❌ 삭제 실패: ${error.message}`, 'error');
    }
}

// ============= 즐겨찾기 =============
async function loadFavorites() {
    try {
        const result = await apiCall('/favorites');
        const list = document.getElementById('favoritesList');
        list.innerHTML = '';
        
        result.favorites.forEach(entry => {
            const item = document.createElement('div');
            item.className = 'favorite-item';
            item.innerHTML = `
                <div class="favorite-item-header">
                    <span class="favorite-item-name">${escapeHtml(entry.name)}</span>
                    <div class="item-actions">
                        <button class="btn btn-secondary" onclick="useFavorite('${escapeHtml(entry.prompt)}')">사용</button>
                        <button class="btn btn-danger" onclick="deleteFavorite('${entry.id}')">삭제</button>
                    </div>
                </div>
                <div class="favorite-item-prompt">${escapeHtml(entry.prompt)}</div>
            `;
            list.appendChild(item);
        });
    } catch (error) {
        console.error('즐겨찾기 로드 실패:', error);
    }
}

function useFavorite(prompt) {
    promptInput.value = prompt;
    const koreanInputEl = document.getElementById('koreanInput');
    if (koreanInputEl) koreanInputEl.value = '';
    switchTab('chat');
    addMessage('system', '✅ 즐겨찾기 적용됨 (영어 프롬프트 직접 사용)');
}

async function saveFavorite() {
    const name = document.getElementById('favNameInput').value.trim();
    const prompt = promptInput.value.trim();
    
    if (!name || !prompt) {
        alert('이름과 프롬프트를 입력해주세요.');
        return;
    }
    
    try {
        await apiCall('/favorites', 'POST', {
            name,
            prompt
        });
        
        closeModal('saveFavoriteModal');
        loadFavorites();
        addMessage('system', `✅ 즐겨찾기 저장: ${name}`);
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

async function deleteFavorite(id) {
    if (!confirm('이 즐겨찾기를 삭제하시겠습니까?')) return;
    
    try {
        await apiCall(`/favorites/${id}`, 'DELETE');
        loadFavorites();
    } catch (error) {
        addMessage('system', `❌ 삭제 실패: ${error.message}`, 'error');
    }
}

// ============= 설정 =============
let llmProviders = {};
let defaultTranslatePrompt = '';
let defaultEnhancePrompt = '';
// 편집 시스템 프롬프트 기본값
let defaultEditTranslatePrompt = '';
let defaultEditEnhancePrompt = '';
let defaultEditSuggestPrompt = '';

async function loadLlmProviders() {
    try {
        const result = await apiCall('/settings');
        llmProviders = result.llm_providers || {};
        
        const currentProvider = result.llm_provider || 'openai';
        const currentModel = result.llm_model || '';
        
        const providerSelect = document.getElementById('llmProviderSelect');
        const chatProviderSelect = document.getElementById('chatLlmProviderSelect');
        
        [providerSelect, chatProviderSelect].forEach(select => {
            if (select) {
                select.innerHTML = '';
                for (const [pid, pinfo] of Object.entries(llmProviders)) {
                    const opt = document.createElement('option');
                    opt.value = pid;
                    opt.textContent = pinfo.name;
                    select.appendChild(opt);
                }
                select.value = currentProvider;
            }
        });
        
        updateLlmModelList(currentProvider, currentModel);
        updateChatLlmModelList(currentProvider, currentModel);
        
        updateLlmBaseUrlVisibility(currentProvider);
        if (result.llm_base_url) {
            const baseUrlInput = document.getElementById('llmBaseUrlInput');
            if (baseUrlInput) baseUrlInput.value = result.llm_base_url;
        }
        
        defaultTranslatePrompt = result.default_translate_system_prompt || '';
        defaultEnhancePrompt = result.default_enhance_system_prompt || '';
        
        const translatePromptInput = document.getElementById('translateSystemPrompt');
        const enhancePromptInput = document.getElementById('enhanceSystemPrompt');
        
        if (translatePromptInput) {
            translatePromptInput.value = result.translate_system_prompt || defaultTranslatePrompt;
        }
        if (enhancePromptInput) {
            enhancePromptInput.value = result.enhance_system_prompt || defaultEnhancePrompt;
        }
        
        // 편집 시스템 프롬프트 기본값 및 현재값 로드
        defaultEditTranslatePrompt = result.default_edit_translate_system_prompt || '';
        defaultEditEnhancePrompt = result.default_edit_enhance_system_prompt || '';
        defaultEditSuggestPrompt = result.default_edit_suggest_system_prompt || '';
        
        const editTranslatePromptInput = document.getElementById('editTranslateSystemPrompt');
        const editEnhancePromptInput = document.getElementById('editEnhanceSystemPrompt');
        const editSuggestPromptInput = document.getElementById('editSuggestSystemPrompt');
        
        if (editTranslatePromptInput) {
            editTranslatePromptInput.value = result.edit_translate_system_prompt || defaultEditTranslatePrompt;
        }
        if (editEnhancePromptInput) {
            editEnhancePromptInput.value = result.edit_enhance_system_prompt || defaultEditEnhancePrompt;
        }
        if (editSuggestPromptInput) {
            editSuggestPromptInput.value = result.edit_suggest_system_prompt || defaultEditSuggestPrompt;
        }
        
        // 관리자 상태 업데이트
        if (result.is_admin !== undefined) {
            isAdmin = result.is_admin;
            updateAdminUI();
        }
        
        console.log('LLM 프로바이더 로드 완료:', Object.keys(llmProviders).length + '개');
    } catch (error) {
        console.error('LLM 프로바이더 로드 실패:', error);
    }
}

function updateLlmModelList(providerId, currentModel = '') {
    const modelSelect = document.getElementById('llmModelSelect');
    const customInput = document.getElementById('llmModelCustomInput');
    
    // 'env' provider는 별도 처리 (updateLlmBaseUrlVisibility에서 처리)
    if (providerId === 'env') return;
    
    if (!modelSelect || !llmProviders[providerId]) return;
    
    const provider = llmProviders[providerId];
    modelSelect.innerHTML = '<option value="">기본 모델</option>';
    
    provider.models.forEach(model => {
        const opt = document.createElement('option');
        opt.value = model;
        opt.textContent = model;
        modelSelect.appendChild(opt);
    });
    
    const customOpt = document.createElement('option');
    customOpt.value = '__custom__';
    customOpt.textContent = '✏️ 직접 입력...';
    modelSelect.appendChild(customOpt);
    
    const isPresetModel = currentModel === '' || provider.models.includes(currentModel);
    
    if (isPresetModel) {
        modelSelect.value = currentModel;
        if (customInput) customInput.style.display = 'none';
    } else {
        modelSelect.value = '__custom__';
        if (customInput) {
            customInput.style.display = 'block';
            customInput.value = currentModel;
        }
    }
    
    const infoEl = document.getElementById('llmProviderInfo');
    if (infoEl) {
        let infoText = `💡 ${provider.name}`;
        if (provider.default_model) {
            infoText += ` - 기본 모델: ${provider.default_model}`;
        }
        if (providerId === 'ollama' || providerId === 'lmstudio') {
            infoText += ' (로컬 서버가 실행 중이어야 합니다)';
        }
        infoEl.innerHTML = `<small>${infoText}</small>`;
    }
}

function updateChatLlmModelList(providerId, currentModel = '') {
    const modelSelect = document.getElementById('chatLlmModelSelect');
    
    // 'env' provider는 모델 목록 비움
    if (providerId === 'env') {
        if (modelSelect) modelSelect.innerHTML = '<option value="">.env 설정</option>';
        return;
    }
    
    if (!modelSelect || !llmProviders[providerId]) return;
    
    const provider = llmProviders[providerId];
    modelSelect.innerHTML = '<option value="">기본</option>';
    
    provider.models.forEach(model => {
        const opt = document.createElement('option');
        opt.value = model;
        opt.textContent = model.length > 20 ? model.substring(0, 18) + '...' : model;
        opt.title = model;
        if (model === currentModel) opt.selected = true;
        modelSelect.appendChild(opt);
    });
}

async function saveChatLlmSettings() {
    if (!isAdmin) return;  // 관리자만 저장 가능
    
    const provider = document.getElementById('chatLlmProviderSelect')?.value;
    const model = document.getElementById('chatLlmModelSelect')?.value;
    
    if (!provider) return;
    
    try {
        await apiCall('/settings', 'POST', {
            llm_provider: provider,
            llm_model: model
        });
        
        const settingsProviderSelect = document.getElementById('llmProviderSelect');
        const settingsModelSelect = document.getElementById('llmModelSelect');
        if (settingsProviderSelect) settingsProviderSelect.value = provider;
        if (settingsModelSelect) {
            updateLlmModelList(provider, model);
        }
        updateLlmBaseUrlVisibility(provider);
        
        addMessage('system', `✅ LLM: ${llmProviders[provider]?.name || provider}${model ? ' / ' + model : ''}`);
    } catch (error) {
        console.error('LLM 설정 저장 실패:', error);
    }
}

function updateLlmBaseUrlVisibility(providerId) {
    const baseUrlGroup = document.getElementById('llmBaseUrlGroup');
    const apiKeyInput = document.getElementById('llmApiKeyInput');
    const modelSelectWrapper = document.querySelector('.model-input-wrapper');
    const infoEl = document.getElementById('llmProviderInfo');
    
    // 'env' provider: 모든 설정 필드 숨기기
    if (providerId === 'env') {
        if (baseUrlGroup) baseUrlGroup.style.display = 'none';
        if (apiKeyInput) apiKeyInput.parentElement.style.display = 'none';
        if (modelSelectWrapper) modelSelectWrapper.parentElement.style.display = 'none';
        if (infoEl) {
            infoEl.innerHTML = '<small>📁 <strong>.env 파일</strong>의 설정을 사용합니다. (LLM_PROVIDER, LLM_API_KEY, LLM_MODEL 등)</small>';
        }
        return;
    }
    
    // 다른 provider: 필드 표시
    if (apiKeyInput) apiKeyInput.parentElement.style.display = 'block';
    if (modelSelectWrapper) modelSelectWrapper.parentElement.style.display = 'block';
    
    if (baseUrlGroup) {
        baseUrlGroup.style.display = 
            (providerId === 'custom' || providerId === 'ollama' || providerId === 'lmstudio') 
            ? 'block' : 'none';
    }
}

async function saveLlmSettings() {
    if (!isAdmin) {
        addMessage('system', '❌ 설정 변경은 관리자만 가능합니다.', 'error');
        return;
    }
    
    const provider = document.getElementById('llmProviderSelect').value;
    const apiKey = document.getElementById('llmApiKeyInput').value.trim();
    const baseUrl = document.getElementById('llmBaseUrlInput').value.trim();
    
    const modelSelect = document.getElementById('llmModelSelect');
    const customInput = document.getElementById('llmModelCustomInput');
    let model = modelSelect.value;
    
    if (model === '__custom__' && customInput) {
        model = customInput.value.trim();
    }
    
    try {
        await apiCall('/settings', 'POST', {
            llm_provider: provider,
            llm_api_key: apiKey,
            llm_base_url: baseUrl,
            llm_model: model
        });
        
        const chatProviderSelect = document.getElementById('chatLlmProviderSelect');
        const chatModelSelect = document.getElementById('chatLlmModelSelect');
        if (chatProviderSelect) chatProviderSelect.value = provider;
        if (chatModelSelect) updateChatLlmModelList(provider, model);
        
        addMessage('system', `✅ LLM 설정 저장됨 (${llmProviders[provider]?.name || provider}${model ? ' / ' + model : ''})`);
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

async function saveApiKey() {
    const apiKey = document.getElementById('apiKeyInput')?.value?.trim() || 
                   document.getElementById('llmApiKeyInput')?.value?.trim();
    
    try {
        await apiCall('/settings', 'POST', { openai_api_key: apiKey });
        addMessage('system', '✅ API 키 저장됨');
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

async function saveAutoUnloadSettings() {
    if (!isAdmin) {
        addMessage('system', '❌ 자동 언로드 설정은 관리자만 변경할 수 있습니다.', 'error');
        return;
    }
    
    const enabled = document.getElementById('autoUnloadEnabledCheck')?.checked ?? true;
    const timeout = parseInt(document.getElementById('autoUnloadTimeoutInput')?.value) || 10;
    
    try {
        await apiCall('/settings', 'POST', {
            auto_unload_enabled: enabled,
            auto_unload_timeout: timeout
        });
        
        const statusText = enabled ? `${timeout}분 후 자동 언로드` : '비활성화';
        addMessage('system', `✅ 자동 언로드 설정 저장됨 (${statusText})`);
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

async function saveEditAutoUnloadSettings() {
    if (!isAdmin) {
        addMessage('system', '❌ 편집 모델 자동 언로드 설정은 관리자만 변경할 수 있습니다.', 'error');
        return;
    }
    
    const enabled = document.getElementById('editAutoUnloadEnabledCheck')?.checked ?? true;
    const timeout = parseInt(document.getElementById('editAutoUnloadTimeoutInput')?.value) || 10;
    
    try {
        await apiCall('/settings', 'POST', {
            edit_auto_unload_enabled: enabled,
            edit_auto_unload_timeout: timeout
        });
        
        const statusText = enabled ? `${timeout}분 후 자동 언로드` : '비활성화';
        addMessage('system', `✅ 편집 모델 자동 언로드 설정 저장됨 (${statusText})`);
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

async function loadAutoUnloadSettings() {
    try {
        const result = await apiCall('/settings');

        // Z-Image 자동 언로드 설정
        const enabledCheck = document.getElementById('autoUnloadEnabledCheck');
        const timeoutInput = document.getElementById('autoUnloadTimeoutInput');

        if (enabledCheck) {
            enabledCheck.checked = result.auto_unload_enabled ?? true;
        }
        if (timeoutInput) {
            timeoutInput.value = result.auto_unload_timeout ?? 10;
        }

        // 편집 모델 자동 언로드 설정
        const editEnabledCheck = document.getElementById('editAutoUnloadEnabledCheck');
        const editTimeoutInput = document.getElementById('editAutoUnloadTimeoutInput');

        if (editEnabledCheck) {
            editEnabledCheck.checked = result.edit_auto_unload_enabled ?? true;
        }
        if (editTimeoutInput) {
            editTimeoutInput.value = result.edit_auto_unload_timeout ?? 10;
        }

        console.log('자동 언로드 설정 로드 완료:', {
            enabled: result.auto_unload_enabled,
            timeout: result.auto_unload_timeout,
            edit_enabled: result.edit_auto_unload_enabled,
            edit_timeout: result.edit_auto_unload_timeout
        });
    } catch (error) {
        console.error('자동 언로드 설정 로드 실패:', error);
    }
}

async function saveSystemPrompts() {
    // 시스템 프롬프트는 세션별 개인화 - 모든 사용자 저장 가능
    const translatePrompt = document.getElementById('translateSystemPrompt')?.value || '';
    const enhancePrompt = document.getElementById('enhanceSystemPrompt')?.value || '';
    
    try {
        await apiCall('/settings/prompts', 'POST', {
            translate_system_prompt: translatePrompt,
            enhance_system_prompt: enhancePrompt
        });
        addMessage('system', '✅ 시스템 프롬프트 저장됨 (내 설정)');
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

async function resetTranslatePrompt() {
    const translatePromptInput = document.getElementById('translateSystemPrompt');
    if (translatePromptInput && defaultTranslatePrompt) {
        translatePromptInput.value = defaultTranslatePrompt;
        
        // 세션 설정에서 삭제하여 기본값 사용
        try {
            await apiCall('/settings/prompts', 'POST', {
                translate_system_prompt: ''  // 빈 문자열로 저장하면 기본값 사용
            });
        } catch (error) {
            console.error('번역 프롬프트 초기화 실패:', error);
        }
        
        addMessage('system', '✅ 번역 시스템 프롬프트가 기본값으로 초기화되었습니다.');
    }
}

async function resetEnhancePrompt() {
    const enhancePromptInput = document.getElementById('enhanceSystemPrompt');
    if (enhancePromptInput && defaultEnhancePrompt) {
        enhancePromptInput.value = defaultEnhancePrompt;
        
        // 세션 설정에서 삭제하여 기본값 사용
        try {
            await apiCall('/settings/prompts', 'POST', {
                enhance_system_prompt: ''  // 빈 문자열로 저장하면 기본값 사용
            });
        } catch (error) {
            console.error('향상 프롬프트 초기화 실패:', error);
        }
        
        addMessage('system', '✅ 향상 시스템 프롬프트가 기본값으로 초기화되었습니다.');
    }
}

// ============= 편집 시스템 프롬프트 설정 (개인화) =============
async function saveEditSystemPrompts() {
    // 편집 시스템 프롬프트는 세션별 개인화 - 모든 사용자 저장 가능
    const editTranslatePrompt = document.getElementById('editTranslateSystemPrompt')?.value || '';
    const editEnhancePrompt = document.getElementById('editEnhanceSystemPrompt')?.value || '';
    const editSuggestPrompt = document.getElementById('editSuggestSystemPrompt')?.value || '';
    
    try {
        await apiCall('/settings/prompts', 'POST', {
            edit_translate_system_prompt: editTranslatePrompt,
            edit_enhance_system_prompt: editEnhancePrompt,
            edit_suggest_system_prompt: editSuggestPrompt
        });
        addMessage('system', '✅ 편집 시스템 프롬프트 저장됨 (내 설정)');
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

async function resetEditTranslatePrompt() {
    const editTranslatePromptInput = document.getElementById('editTranslateSystemPrompt');
    if (editTranslatePromptInput && defaultEditTranslatePrompt) {
        editTranslatePromptInput.value = defaultEditTranslatePrompt;
        
        // 세션 설정에서 삭제하여 기본값 사용
        try {
            await apiCall('/settings/prompts', 'POST', {
                edit_translate_system_prompt: ''
            });
        } catch (error) {
            console.error('편집 번역 프롬프트 초기화 실패:', error);
        }
        
        addMessage('system', '✅ 편집 지시어 번역 프롬프트가 기본값으로 초기화되었습니다.');
    }
}

async function resetEditEnhancePrompt() {
    const editEnhancePromptInput = document.getElementById('editEnhanceSystemPrompt');
    if (editEnhancePromptInput && defaultEditEnhancePrompt) {
        editEnhancePromptInput.value = defaultEditEnhancePrompt;
        
        // 세션 설정에서 삭제하여 기본값 사용
        try {
            await apiCall('/settings/prompts', 'POST', {
                edit_enhance_system_prompt: ''
            });
        } catch (error) {
            console.error('편집 향상 프롬프트 초기화 실패:', error);
        }
        
        addMessage('system', '✅ 편집 지시어 향상 프롬프트가 기본값으로 초기화되었습니다.');
    }
}

async function resetEditSuggestPrompt() {
    const editSuggestPromptInput = document.getElementById('editSuggestSystemPrompt');
    if (editSuggestPromptInput && defaultEditSuggestPrompt) {
        editSuggestPromptInput.value = defaultEditSuggestPrompt;
        
        // 세션 설정에서 삭제하여 기본값 사용
        try {
            await apiCall('/settings/prompts', 'POST', {
                edit_suggest_system_prompt: ''
            });
        } catch (error) {
            console.error('편집 제안 프롬프트 초기화 실패:', error);
        }
        
        addMessage('system', '✅ 편집 제안 프롬프트가 기본값으로 초기화되었습니다.');
    }
}

// ============= UI 헬퍼 =============
function switchTab(tabId) {
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.toggle('active', tab.id === `tab-${tabId}`);
    });
    
    if (tabId === 'gallery') loadGallery();
    if (tabId === 'history') loadHistory();
    if (tabId === 'favorites') loadFavorites();
    if (tabId === 'settings' && isAdmin) {
        loadSessionList();
        loadAdminGpuPanel();
    }
    if (tabId === 'edit-history') loadEditHistory();
    if (tabId === 'edit') loadEditQuantizationOptions();
}

// ============= 이미지 미리보기 (줌/드래그/네비게이션 지원) =============
let imagePreviewState = {
    scale: 1,
    translateX: 0,
    translateY: 0,
    isDragging: false,
    startX: 0,
    startY: 0,
    lastTranslateX: 0,
    lastTranslateY: 0,
    currentPath: '',
    fitMode: true,
    naturalWidth: 0,
    naturalHeight: 0,
    // 이미지 목록 네비게이션
    imageList: [],      // [{path, metadata}, ...]
    currentIndex: 0
};

// 단일 이미지 보기 (기존 호환)
function showImageModal(path, metadata) {
    showImageModalWithList([{path, metadata}], 0);
}

// 이미지 목록과 함께 보기 (네비게이션 지원)
function showImageModalWithList(imageList, startIndex = 0) {
    const modal = document.getElementById('imageModal');
    const wrapper = document.getElementById('imagePreviewWrapper');
    
    // 상태 초기화
    imagePreviewState = {
        scale: 1,
        translateX: 0,
        translateY: 0,
        isDragging: false,
        startX: 0,
        startY: 0,
        lastTranslateX: 0,
        lastTranslateY: 0,
        currentPath: '',
        fitMode: true,
        naturalWidth: 0,
        naturalHeight: 0,
        imageList: imageList,
        currentIndex: startIndex
    };
    
    wrapper.classList.add('fit-mode');
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    
    // 현재 이미지 표시
    showCurrentImage();
    updateNavigationButtons();
    updateImageCounter();
}

// 현재 인덱스의 이미지 표시
function showCurrentImage() {
    const img = document.getElementById('modalImage');
    const info = document.getElementById('modalInfo');
    
    if (imagePreviewState.imageList.length === 0) return;
    
    const current = imagePreviewState.imageList[imagePreviewState.currentIndex];
    const path = current.path;
    const metadata = current.metadata;
    
    // 줌/이동 상태 초기화
    imagePreviewState.scale = 1;
    imagePreviewState.translateX = 0;
    imagePreviewState.translateY = 0;
    imagePreviewState.fitMode = true;
    imagePreviewState.currentPath = path;
    
    // 이미지 로드 후 원본 크기 저장
    img.onload = () => {
        imagePreviewState.naturalWidth = img.naturalWidth;
        imagePreviewState.naturalHeight = img.naturalHeight;
        updateImageTransform();
        updateZoomLevel();
    };
    
    img.src = path;
    updateImageTransform();
    updateZoomLevel();
    
    // 메타데이터 표시
    if (metadata) {
        let infoText = '';
        if (metadata.prompt) infoText += `📝 프롬프트: ${metadata.prompt}\n`;
        if (metadata.seed) infoText += `🎲 시드: ${metadata.seed}\n`;
        if (metadata.width && metadata.height) infoText += `📐 해상도: ${metadata.width}×${metadata.height}\n`;
        if (metadata.steps) infoText += `🔄 스텝: ${metadata.steps}\n`;
        info.textContent = infoText;
    } else {
        info.textContent = '';
    }
}

// 이전 이미지
function showPrevImage() {
    if (imagePreviewState.currentIndex > 0) {
        imagePreviewState.currentIndex--;
        showCurrentImage();
        updateNavigationButtons();
        updateImageCounter();
    }
}

// 다음 이미지
function showNextImage() {
    if (imagePreviewState.currentIndex < imagePreviewState.imageList.length - 1) {
        imagePreviewState.currentIndex++;
        showCurrentImage();
        updateNavigationButtons();
        updateImageCounter();
    }
}

// 네비게이션 버튼 상태 업데이트
function updateNavigationButtons() {
    const prevBtn = document.getElementById('btnPrevImage');
    const nextBtn = document.getElementById('btnNextImage');
    const total = imagePreviewState.imageList.length;
    
    if (total <= 1) {
        // 이미지가 1개면 버튼 숨김
        prevBtn.style.display = 'none';
        nextBtn.style.display = 'none';
    } else {
        prevBtn.style.display = 'flex';
        nextBtn.style.display = 'flex';
        prevBtn.disabled = imagePreviewState.currentIndex === 0;
        nextBtn.disabled = imagePreviewState.currentIndex === total - 1;
    }
}

// 이미지 카운터 업데이트
function updateImageCounter() {
    const counter = document.getElementById('imageCounter');
    const total = imagePreviewState.imageList.length;
    
    if (total <= 1) {
        counter.classList.remove('visible');
    } else {
        counter.innerHTML = `<span class="current">${imagePreviewState.currentIndex + 1}</span> / ${total}`;
        counter.classList.add('visible');
    }
}

function updateImageTransform() {
    const img = document.getElementById('modalImage');
    const wrapper = document.getElementById('imagePreviewWrapper');
    
    if (imagePreviewState.fitMode) {
        // 화면에 맞춤 모드: CSS로 크기 제한
        wrapper.classList.add('fit-mode');
        img.style.transform = '';
        img.style.maxWidth = '95vw';
        img.style.maxHeight = 'calc(100vh - 160px)';
        img.style.width = 'auto';
        img.style.height = 'auto';
    } else {
        // 자유 줌/이동 모드: 제한 해제하고 transform 적용
        wrapper.classList.remove('fit-mode');
        img.style.maxWidth = 'none';
        img.style.maxHeight = 'none';
        
        // 원본 크기 기준으로 표시
        if (imagePreviewState.naturalWidth && imagePreviewState.naturalHeight) {
            img.style.width = imagePreviewState.naturalWidth + 'px';
            img.style.height = imagePreviewState.naturalHeight + 'px';
        }
        
        img.style.transform = `translate(${imagePreviewState.translateX}px, ${imagePreviewState.translateY}px) scale(${imagePreviewState.scale})`;
    }
}

function updateZoomLevel() {
    const zoomEl = document.getElementById('zoomLevel');
    if (zoomEl) {
        zoomEl.textContent = `${Math.round(imagePreviewState.scale * 100)}%`;
    }
}

function zoomImage(delta) {
    const wrapper = document.getElementById('imagePreviewWrapper');
    
    // 맞춤 모드에서 줌 시작하면 현재 크기 기준으로 전환
    if (imagePreviewState.fitMode) {
        imagePreviewState.fitMode = false;
        imagePreviewState.scale = 1;
        imagePreviewState.translateX = 0;
        imagePreviewState.translateY = 0;
    }
    
    const newScale = Math.max(0.1, Math.min(10, imagePreviewState.scale + delta));
    imagePreviewState.scale = newScale;
    
    updateImageTransform();
    updateZoomLevel();
}

function zoomToFit() {
    const img = document.getElementById('modalImage');
    
    imagePreviewState.fitMode = true;
    imagePreviewState.scale = 1;
    imagePreviewState.translateX = 0;
    imagePreviewState.translateY = 0;
    
    // 인라인 스타일 초기화
    img.style.width = 'auto';
    img.style.height = 'auto';
    
    updateImageTransform();
    updateZoomLevel();
}

function zoomToOriginal() {
    const img = document.getElementById('modalImage');
    
    // 원본 픽셀 크기로 표시 (1:1)
    imagePreviewState.fitMode = false;
    imagePreviewState.scale = 1;
    imagePreviewState.translateX = 0;
    imagePreviewState.translateY = 0;
    
    // 이미지 크기를 원본 픽셀 크기로 명시적 설정
    if (imagePreviewState.naturalWidth && imagePreviewState.naturalHeight) {
        img.style.width = imagePreviewState.naturalWidth + 'px';
        img.style.height = imagePreviewState.naturalHeight + 'px';
    }
    
    updateImageTransform();
    updateZoomLevel();
}

function downloadPreviewImage() {
    if (imagePreviewState.currentPath) {
        const link = document.createElement('a');
        link.href = imagePreviewState.currentPath;
        link.download = imagePreviewState.currentPath.split('/').pop() || 'image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

// 여러 이미지를 ZIP으로 묶어서 다운로드
async function downloadImagesAsZip(images, prompt) {
    // JSZip 라이브러리 확인
    if (typeof JSZip === 'undefined') {
        alert('ZIP 라이브러리를 불러오지 못했습니다. 페이지를 새로고침해주세요.');
        return;
    }
    
    const zip = new JSZip();
    const folder = zip.folder('images');
    
    // 다운로드 진행 상태 표시
    const statusMsg = addMessage('system', `📦 ${images.length}장의 이미지를 다운로드 준비 중...`);
    
    try {
        // 각 이미지를 fetch하여 ZIP에 추가
        const fetchPromises = images.map(async (img, index) => {
            try {
                const response = await fetch(img.path);
                const blob = await response.blob();
                
                // 파일명 생성 (시드 포함)
                const filename = img.path.split('/').pop() || `image_${index + 1}_seed${img.seed}.png`;
                folder.file(filename, blob);
                
                return true;
            } catch (error) {
                console.error(`이미지 ${index + 1} 다운로드 실패:`, error);
                return false;
            }
        });
        
        const results = await Promise.all(fetchPromises);
        const successCount = results.filter(r => r).length;
        
        if (successCount === 0) {
            updateMessageText(statusMsg, '❌ 이미지 다운로드에 실패했습니다.');
            return;
        }
        
        // ZIP 파일 생성
        updateMessageText(statusMsg, `📦 ZIP 파일 생성 중... (${successCount}/${images.length}장)`);
        
        const content = await zip.generateAsync({ type: 'blob' });
        
        // 파일명 생성 (날짜 + 프롬프트 일부)
        const date = new Date();
        const dateStr = `${date.getFullYear()}${(date.getMonth()+1).toString().padStart(2,'0')}${date.getDate().toString().padStart(2,'0')}`;
        const timeStr = `${date.getHours().toString().padStart(2,'0')}${date.getMinutes().toString().padStart(2,'0')}`;
        const promptShort = prompt.substring(0, 30).replace(/[^a-zA-Z0-9]/g, '_');
        const zipFilename = `images_${dateStr}_${timeStr}_${promptShort}.zip`;
        
        // 다운로드 링크 생성
        const url = URL.createObjectURL(content);
        const link = document.createElement('a');
        link.href = url;
        link.download = zipFilename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        updateMessageText(statusMsg, `✅ ${successCount}장의 이미지가 ZIP 파일로 다운로드되었습니다.`);
        
    } catch (error) {
        console.error('ZIP 생성 실패:', error);
        updateMessageText(statusMsg, `❌ ZIP 생성 실패: ${error.message}`);
    }
}

// 메시지 텍스트 업데이트 헬퍼
function updateMessageText(messageEl, text) {
    if (messageEl) {
        const contentEl = messageEl.querySelector('.message-content p');
        if (contentEl) {
            contentEl.textContent = text;
        }
    }
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

// 이미지 드래그 이벤트
function initImagePreviewDrag() {
    const wrapper = document.getElementById('imagePreviewWrapper');
    if (!wrapper) return;
    
    wrapper.addEventListener('mousedown', (e) => {
        if (imagePreviewState.fitMode) return;
        
        imagePreviewState.isDragging = true;
        imagePreviewState.startX = e.clientX;
        imagePreviewState.startY = e.clientY;
        imagePreviewState.lastTranslateX = imagePreviewState.translateX;
        imagePreviewState.lastTranslateY = imagePreviewState.translateY;
        wrapper.style.cursor = 'grabbing';
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!imagePreviewState.isDragging) return;
        
        const deltaX = e.clientX - imagePreviewState.startX;
        const deltaY = e.clientY - imagePreviewState.startY;
        
        imagePreviewState.translateX = imagePreviewState.lastTranslateX + deltaX;
        imagePreviewState.translateY = imagePreviewState.lastTranslateY + deltaY;
        
        updateImageTransform();
    });
    
    document.addEventListener('mouseup', () => {
        if (imagePreviewState.isDragging) {
            imagePreviewState.isDragging = false;
            const wrapper = document.getElementById('imagePreviewWrapper');
            if (wrapper) wrapper.style.cursor = 'grab';
        }
    });
    
    // 마우스 휠 줌
    wrapper.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        zoomImage(delta);
    }, { passive: false });
    
    // 더블클릭으로 맞춤/원본 토글
    wrapper.addEventListener('dblclick', () => {
        if (imagePreviewState.fitMode) {
            zoomToOriginal();
        } else {
            zoomToFit();
        }
    });
}

// 터치 지원
function initImagePreviewTouch() {
    const wrapper = document.getElementById('imagePreviewWrapper');
    if (!wrapper) return;
    
    let initialDistance = 0;
    let initialScale = 1;
    
    wrapper.addEventListener('touchstart', (e) => {
        if (e.touches.length === 2) {
            initialDistance = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY
            );
            initialScale = imagePreviewState.scale;
            if (imagePreviewState.fitMode) {
                imagePreviewState.fitMode = false;
                imagePreviewState.scale = 1;
                initialScale = 1;
            }
        } else if (e.touches.length === 1 && !imagePreviewState.fitMode) {
            imagePreviewState.isDragging = true;
            imagePreviewState.startX = e.touches[0].clientX;
            imagePreviewState.startY = e.touches[0].clientY;
            imagePreviewState.lastTranslateX = imagePreviewState.translateX;
            imagePreviewState.lastTranslateY = imagePreviewState.translateY;
        }
    }, { passive: true });
    
    wrapper.addEventListener('touchmove', (e) => {
        if (e.touches.length === 2) {
            const currentDistance = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY
            );
            const scaleChange = currentDistance / initialDistance;
            imagePreviewState.scale = Math.max(0.1, Math.min(10, initialScale * scaleChange));
            updateImageTransform();
            updateZoomLevel();
        } else if (e.touches.length === 1 && imagePreviewState.isDragging) {
            const deltaX = e.touches[0].clientX - imagePreviewState.startX;
            const deltaY = e.touches[0].clientY - imagePreviewState.startY;
            imagePreviewState.translateX = imagePreviewState.lastTranslateX + deltaX;
            imagePreviewState.translateY = imagePreviewState.lastTranslateY + deltaY;
            updateImageTransform();
        }
    }, { passive: true });
    
    wrapper.addEventListener('touchend', () => {
        imagePreviewState.isDragging = false;
    });
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('ko-KR', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============= 이벤트 리스너 =============
document.addEventListener('DOMContentLoaded', () => {
    // 현재 사용자 정보 로드
    loadCurrentUser();
    
    // WebSocket 연결
    connectWebSocket();
    
    // 초기 데이터 로드
    updateModelStatus();
    loadTemplates();
    loadQuantizationOptions();
    // 설정 탭에서도 편집 모델 양자화 옵션을 로드하여 저장값을 즉시 반영
    loadEditQuantizationOptions();
    loadLlmProviders();
    loadAutoUnloadSettings();
    
    // 탭 전환
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
    
    // 생성 버튼
    document.getElementById('btnGenerate').addEventListener('click', () => generateImage(false));
    document.getElementById('btnPreview').addEventListener('click', () => generateImage(true));
    
    // 프롬프트 도구
    document.getElementById('btnTranslate').addEventListener('click', translatePrompt);
    document.getElementById('btnEnhance').addEventListener('click', enhancePrompt);
    document.getElementById('btnTemplate').addEventListener('click', () => {
        document.getElementById('templateModal').classList.add('active');
    });
    
    // 한국어 입력창 번역 버튼
    const btnTranslateKorean = document.getElementById('btnTranslateKorean');
    if (btnTranslateKorean) {
        btnTranslateKorean.addEventListener('click', translateKoreanInput);
    }
    
    // 한국어 입력창 이벤트
    const koreanInputEl = document.getElementById('koreanInput');
    if (koreanInputEl) {
        koreanInputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                generateImage(false);
            }
        });
    }
    
    // 모델 관리 - 설정 탭
    document.getElementById('btnLoadModel').addEventListener('click', () => loadModel(false));
    document.getElementById('btnUnloadModel').addEventListener('click', unloadModel);
    
    // 모델 관리 - 대화 탭
    document.getElementById('btnChatLoadModel').addEventListener('click', () => loadModel(true));
    document.getElementById('btnChatUnloadModel').addEventListener('click', unloadModel);

    // 모델 설정(양자화/CPU 오프로딩) 변경 시 저장 (관리자만)
    const quantizationSelect = document.getElementById('quantizationSelect');
    const cpuOffloadCheck = document.getElementById('cpuOffloadCheck');
    const editQuantizationSelectSettings = document.getElementById('editQuantizationSelectSettings');
    const editCpuOffloadCheckSettings = document.getElementById('editCpuOffloadCheckSettings');

    [quantizationSelect, cpuOffloadCheck, editQuantizationSelectSettings, editCpuOffloadCheckSettings].forEach(el => {
        if (!el) return;
        el.addEventListener('change', () => {
            if (isAdmin) saveModelSettings();
        });
    });
    
    // 커스텀 해상도 토글
    document.getElementById('resolutionSelect').addEventListener('change', (e) => {
        const customDiv = document.getElementById('customResolution');
        customDiv.style.display = e.target.value === 'custom' ? 'flex' : 'none';
    });
    
    // LLM 설정 - 설정탭
    const llmProviderSelect = document.getElementById('llmProviderSelect');
    if (llmProviderSelect) {
        llmProviderSelect.addEventListener('change', (e) => {
            updateLlmModelList(e.target.value);
            updateLlmBaseUrlVisibility(e.target.value);
        });
    }
    
    // LLM 모델 선택 - 직접 입력 토글
    const llmModelSelect = document.getElementById('llmModelSelect');
    const llmModelCustomInput = document.getElementById('llmModelCustomInput');
    if (llmModelSelect && llmModelCustomInput) {
        llmModelSelect.addEventListener('change', (e) => {
            if (e.target.value === '__custom__') {
                llmModelCustomInput.style.display = 'block';
                llmModelCustomInput.focus();
            } else {
                llmModelCustomInput.style.display = 'none';
                llmModelCustomInput.value = '';
            }
        });
    }
    
    // LLM 설정 - 대화탭 (빠른 선택)
    const chatLlmProviderSelect = document.getElementById('chatLlmProviderSelect');
    if (chatLlmProviderSelect) {
        chatLlmProviderSelect.addEventListener('change', (e) => {
            updateChatLlmModelList(e.target.value);
            if (isAdmin) saveChatLlmSettings();
        });
    }
    
    const chatLlmModelSelect = document.getElementById('chatLlmModelSelect');
    if (chatLlmModelSelect) {
        chatLlmModelSelect.addEventListener('change', () => {
            if (isAdmin) saveChatLlmSettings();
        });
    }
    
    const btnSaveLlmSettings = document.getElementById('btnSaveLlmSettings');
    if (btnSaveLlmSettings) {
        btnSaveLlmSettings.addEventListener('click', saveLlmSettings);
    }
    
    // 시스템 프롬프트 설정
    const btnSaveSystemPrompts = document.getElementById('btnSaveSystemPrompts');
    if (btnSaveSystemPrompts) {
        btnSaveSystemPrompts.addEventListener('click', saveSystemPrompts);
    }
    
    const btnResetTranslatePrompt = document.getElementById('btnResetTranslatePrompt');
    if (btnResetTranslatePrompt) {
        btnResetTranslatePrompt.addEventListener('click', resetTranslatePrompt);
    }
    
    const btnResetEnhancePrompt = document.getElementById('btnResetEnhancePrompt');
    if (btnResetEnhancePrompt) {
        btnResetEnhancePrompt.addEventListener('click', resetEnhancePrompt);
    }
    
    // 편집 시스템 프롬프트 설정 (개인화)
    const btnSaveEditSystemPrompts = document.getElementById('btnSaveEditSystemPrompts');
    if (btnSaveEditSystemPrompts) {
        btnSaveEditSystemPrompts.addEventListener('click', saveEditSystemPrompts);
    }
    
    const btnResetEditTranslatePrompt = document.getElementById('btnResetEditTranslatePrompt');
    if (btnResetEditTranslatePrompt) {
        btnResetEditTranslatePrompt.addEventListener('click', resetEditTranslatePrompt);
    }
    
    const btnResetEditEnhancePrompt = document.getElementById('btnResetEditEnhancePrompt');
    if (btnResetEditEnhancePrompt) {
        btnResetEditEnhancePrompt.addEventListener('click', resetEditEnhancePrompt);
    }
    
    const btnResetEditSuggestPrompt = document.getElementById('btnResetEditSuggestPrompt');
    if (btnResetEditSuggestPrompt) {
        btnResetEditSuggestPrompt.addEventListener('click', resetEditSuggestPrompt);
    }
    
    // 자동 언로드 설정
    const btnSaveAutoUnload = document.getElementById('btnSaveAutoUnload');
    if (btnSaveAutoUnload) {
        btnSaveAutoUnload.addEventListener('click', saveAutoUnloadSettings);
    }
    
    // 편집 모델 자동 언로드 설정
    const btnSaveEditAutoUnload = document.getElementById('btnSaveEditAutoUnload');
    if (btnSaveEditAutoUnload) {
        btnSaveEditAutoUnload.addEventListener('click', saveEditAutoUnloadSettings);
    }

    // GPU 관리 (관리자 전용)
    const btnRefreshGpuStatus = document.getElementById('btnRefreshGpuStatus');
    if (btnRefreshGpuStatus) {
        btnRefreshGpuStatus.addEventListener('click', loadAdminGpuPanel);
    }
    const btnSaveGpuSettings = document.getElementById('btnSaveGpuSettings');
    if (btnSaveGpuSettings) {
        btnSaveGpuSettings.addEventListener('click', saveAdminGpuSettings);
    }
    
    // 설정 탭 편집 모델 로드/언로드
    const btnLoadEditModelSettings = document.getElementById('btnLoadEditModelSettings');
    const btnUnloadEditModelSettings = document.getElementById('btnUnloadEditModelSettings');
    if (btnLoadEditModelSettings) {
        btnLoadEditModelSettings.addEventListener('click', async () => {
            await loadEditModel();
        });
    }
    if (btnUnloadEditModelSettings) {
        btnUnloadEditModelSettings.addEventListener('click', unloadEditModel);
    }
    
    // 레거시 호환
    const btnSaveApiKey = document.getElementById('btnSaveApiKey');
    if (btnSaveApiKey) {
        btnSaveApiKey.addEventListener('click', saveApiKey);
    }
    
    // 갤러리
    document.getElementById('btnRefreshGallery').addEventListener('click', loadGallery);
    
    // 갤러리 다운로드 드롭다운
    document.getElementById('btnGalleryDownload').addEventListener('click', toggleGalleryDropdown);
    document.getElementById('btnDownloadAll').addEventListener('click', downloadAllGalleryImages);
    
    // 드롭다운 외부 클릭 시 닫기
    document.addEventListener('click', (e) => {
        const dropdown = document.getElementById('galleryDownloadDropdown');
        if (dropdown && !dropdown.contains(e.target)) {
            dropdown.classList.remove('open');
        }
    });
    
    // 히스토리
    document.getElementById('btnClearHistory').addEventListener('click', clearHistory);
    
    // 즐겨찾기
    document.getElementById('btnSaveFavorite').addEventListener('click', saveFavorite);
    
    // 세션 관리 새로고침
    const btnRefreshSessions = document.getElementById('btnRefreshSessions');
    if (btnRefreshSessions) {
        btnRefreshSessions.addEventListener('click', loadSessionList);
    }
    
    // ============= 인증 관련 이벤트 =============
    // 로그아웃 버튼
    const btnLogout = document.getElementById('btnLogout');
    if (btnLogout) {
        btnLogout.addEventListener('click', logout);
    }
    
    // 비밀번호 변경 버튼
    const btnChangePassword = document.getElementById('btnChangePassword');
    if (btnChangePassword) {
        btnChangePassword.addEventListener('click', openChangePasswordModal);
    }
    
    // 비밀번호 변경 폼
    const changePasswordForm = document.getElementById('changePasswordForm');
    if (changePasswordForm) {
        changePasswordForm.addEventListener('submit', handleChangePassword);
    }
    
    // 관리자: 사용자 목록 새로고침
    const btnRefreshUsers = document.getElementById('btnRefreshUsers');
    if (btnRefreshUsers) {
        btnRefreshUsers.addEventListener('click', loadUserList);
    }
    
    // 이미지 미리보기 모달 이벤트 설정
    initImagePreviewDrag();
    initImagePreviewTouch();
    
    // 모달 닫기
    document.getElementById('closeImageModal').addEventListener('click', closeImageModal);
    document.getElementById('imagePreviewBackdrop').addEventListener('click', closeImageModal);
    
    // 줌 컨트롤
    document.getElementById('btnZoomIn').addEventListener('click', () => zoomImage(0.25));
    document.getElementById('btnZoomOut').addEventListener('click', () => zoomImage(-0.25));
    document.getElementById('btnZoomFit').addEventListener('click', zoomToFit);
    document.getElementById('btnZoomOriginal').addEventListener('click', zoomToOriginal);
    document.getElementById('btnDownloadImage').addEventListener('click', downloadPreviewImage);
    document.getElementById('btnEditThisImage').addEventListener('click', editCurrentImage);
    
    // 이미지 네비게이션 버튼
    document.getElementById('btnPrevImage').addEventListener('click', (e) => {
        e.stopPropagation();
        showPrevImage();
    });
    document.getElementById('btnNextImage').addEventListener('click', (e) => {
        e.stopPropagation();
        showNextImage();
    });
    
    // 키보드 단축키
    document.addEventListener('keydown', (e) => {
        const modal = document.getElementById('imageModal');
        if (!modal.classList.contains('active')) return;
        
        switch(e.key) {
            case 'Escape':
                closeImageModal();
                break;
            case '+':
            case '=':
                zoomImage(0.25);
                break;
            case '-':
                zoomImage(-0.25);
                break;
            case '0':
                zoomToFit();
                break;
            case '1':
                zoomToOriginal();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                showPrevImage();
                break;
            case 'ArrowRight':
                e.preventDefault();
                showNextImage();
                break;
        }
    });
    
    // 모달 외부 클릭 닫기
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    });
    
    // Enter 키로 생성
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generateImage(false);
        }
    });
    
    // 즐겨찾기 저장 버튼
    const favBtn = document.createElement('button');
    favBtn.className = 'option-btn';
    favBtn.innerHTML = '<i class="ri-star-line"></i>';
    favBtn.title = '즐겨찾기 저장';
    favBtn.onclick = () => {
        if (promptInput.value.trim()) {
            document.getElementById('saveFavoriteModal').classList.add('active');
        }
    };
    document.querySelector('.input-options').appendChild(favBtn);
    
    // ============= 편집 탭 이벤트 =============
    initEditTab();
});


// ============= 편집 탭 관련 변수 =============
let isEditModelLoading = false;
let isEditing = false;
let editImageFile = null;
let referenceImageFile = null;


// ============= 편집 탭 초기화 =============
function initEditTab() {
    // 이미지 업로드 영역
    const editImageUpload = document.getElementById('editImageUpload');
    const editImageInput = document.getElementById('editImageInput');
    const referenceImageBox = document.getElementById('referenceImageBox');
    const referenceImageInput = document.getElementById('referenceImageInput');
    
    // 메인 이미지 업로드
    if (editImageUpload) {
        editImageUpload.addEventListener('click', (e) => {
            if (!e.target.closest('.btn') && !e.target.closest('.upload-preview')) {
                editImageInput.click();
            }
        });
        
        editImageUpload.addEventListener('dragover', (e) => {
            e.preventDefault();
            editImageUpload.classList.add('dragover');
        });
        
        editImageUpload.addEventListener('dragleave', () => {
            editImageUpload.classList.remove('dragover');
        });
        
        editImageUpload.addEventListener('drop', (e) => {
            e.preventDefault();
            editImageUpload.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleEditImageUpload(e.dataTransfer.files[0]);
            }
        });
    }
    
    if (editImageInput) {
        editImageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleEditImageUpload(e.target.files[0]);
            }
        });
    }
    
    // 참조 이미지 업로드
    if (referenceImageBox) {
        referenceImageBox.addEventListener('click', (e) => {
            if (!e.target.closest('.btn') && !e.target.closest('.upload-preview')) {
                referenceImageInput.click();
            }
        });
        
        referenceImageBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            referenceImageBox.classList.add('dragover');
        });
        
        referenceImageBox.addEventListener('dragleave', () => {
            referenceImageBox.classList.remove('dragover');
        });
        
        referenceImageBox.addEventListener('drop', (e) => {
            e.preventDefault();
            referenceImageBox.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleReferenceImageUpload(e.dataTransfer.files[0]);
            }
        });
    }
    
    if (referenceImageInput) {
        referenceImageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleReferenceImageUpload(e.target.files[0]);
            }
        });
    }
    
    // 이미지 제거 버튼
    const btnRemoveEditImage = document.getElementById('btnRemoveEditImage');
    if (btnRemoveEditImage) {
        btnRemoveEditImage.addEventListener('click', (e) => {
            e.stopPropagation();
            removeEditImage();
        });
    }
    
    const btnRemoveRefImage = document.getElementById('btnRemoveRefImage');
    if (btnRemoveRefImage) {
        btnRemoveRefImage.addEventListener('click', (e) => {
            e.stopPropagation();
            removeReferenceImage();
        });
    }
    
    // 모델 로드/언로드
    const btnEditLoadModel = document.getElementById('btnEditLoadModel');
    const btnEditUnloadModel = document.getElementById('btnEditUnloadModel');
    
    if (btnEditLoadModel) {
        btnEditLoadModel.addEventListener('click', loadEditModel);
    }
    if (btnEditUnloadModel) {
        btnEditUnloadModel.addEventListener('click', unloadEditModel);
    }
    
    // 편집 버튼
    const btnEdit = document.getElementById('btnEdit');
    if (btnEdit) {
        btnEdit.addEventListener('click', executeEdit);
    }
    
    // 번역/향상 버튼
    const btnEditTranslate = document.getElementById('btnEditTranslate');
    const btnEditEnhance = document.getElementById('btnEditEnhance');
    const btnEditSuggest = document.getElementById('btnEditSuggest');
    const btnEditTranslateKorean = document.getElementById('btnEditTranslateKorean');
    
    if (btnEditTranslate) {
        btnEditTranslate.addEventListener('click', translateEditPrompt);
    }
    if (btnEditEnhance) {
        btnEditEnhance.addEventListener('click', enhanceEditPrompt);
    }
    if (btnEditSuggest) {
        btnEditSuggest.addEventListener('click', suggestEdits);
    }
    if (btnEditTranslateKorean) {
        btnEditTranslateKorean.addEventListener('click', translateEditKoreanInput);
    }
    
    // 편집 히스토리 삭제 버튼
    const btnClearEditHistory = document.getElementById('btnClearEditHistory');
    if (btnClearEditHistory) {
        btnClearEditHistory.addEventListener('click', clearEditHistory);
    }
    
    // 한국어 입력 엔터키
    const editKoreanInput = document.getElementById('editKoreanInput');
    if (editKoreanInput) {
        editKoreanInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                executeEdit();
            }
        });
    }
    
    // 영어 입력 엔터키
    const editPromptInput = document.getElementById('editPromptInput');
    if (editPromptInput) {
        editPromptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                executeEdit();
            }
        });
    }
    
    // 양자화 옵션 로드
    loadEditQuantizationOptions();
    
    // 갤러리에서 선택 버튼
    const btnSelectFromGallery = document.getElementById('btnSelectFromGallery');
    if (btnSelectFromGallery) {
        btnSelectFromGallery.addEventListener('click', (e) => {
            e.stopPropagation();
            openGallerySelectModal();
        });
    }
}


// ============= 이미지 업로드 처리 =============
function handleEditImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        addEditMessage('system', '❌ 이미지 파일만 업로드할 수 있습니다.');
        return;
    }
    
    editImageFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('editUploadPreview');
        const placeholder = document.getElementById('editUploadPlaceholder');
        const img = document.getElementById('editPreviewImage');
        
        img.src = e.target.result;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function handleReferenceImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        addEditMessage('system', '❌ 이미지 파일만 업로드할 수 있습니다.');
        return;
    }
    
    referenceImageFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('referencePreview');
        const placeholder = document.getElementById('referencePlaceholder');
        const img = document.getElementById('referencePreviewImage');
        
        img.src = e.target.result;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function removeEditImage() {
    editImageFile = null;
    
    const preview = document.getElementById('editUploadPreview');
    const placeholder = document.getElementById('editUploadPlaceholder');
    const input = document.getElementById('editImageInput');
    
    preview.style.display = 'none';
    placeholder.style.display = 'flex';
    input.value = '';
}

function removeReferenceImage() {
    referenceImageFile = null;
    
    const preview = document.getElementById('referencePreview');
    const placeholder = document.getElementById('referencePlaceholder');
    const input = document.getElementById('referenceImageInput');
    
    preview.style.display = 'none';
    placeholder.style.display = 'flex';
    input.value = '';
}


// ============= 편집 모델 관리 =============
async function loadEditModel() {
    if (isEditModelLoading) {
        addEditMessage('system', '⚠️ 이미 모델 로딩 중입니다.');
        return;
    }
    
    // 편집 모델 양자화/CPU 오프로딩 설정은 설정 탭에서만 관리
    const quantization = document.getElementById('editQuantizationSelectSettings')?.value || "BF16 (기본, 최고품질)";
    const cpuOffload = document.getElementById('editCpuOffloadCheckSettings')?.checked ?? true;
    
    try {
        setEditModelLoadingState(true);
        addEditMessage('system', '🔄 편집 모델 로딩을 시작합니다...');
        showEditProgress('모델 로딩 준비 중...', 5);

        const targetDevice = isAdmin ? (adminGpuSettings.edit_gpu || 'auto') : 'auto';
        await apiCall('/edit/model/load', 'POST', {
            quantization,
            cpu_offload: cpuOffload,
            target_device: targetDevice
        });
        
        updateEditModelStatus();
        
    } catch (error) {
        addEditMessage('system', `❌ 모델 로드 실패: ${error.message}`, 'error');
        hideEditProgress();
    } finally {
        setEditModelLoadingState(false);
    }
}

async function unloadEditModel() {
    if (isEditModelLoading) {
        addEditMessage('system', '⚠️ 모델 로딩 중에는 언로드할 수 없습니다.');
        return;
    }
    
    try {
        setEditModelLoadingState(true);
        showEditProgress('모델 언로드 중...', 50);
        addEditMessage('system', '🔄 편집 모델 언로드 중...');
        
        await apiCall('/edit/model/unload', 'POST');
        updateEditModelStatus();
        
    } catch (error) {
        addEditMessage('system', `❌ 모델 언로드 실패: ${error.message}`, 'error');
        hideEditProgress();
    } finally {
        setEditModelLoadingState(false);
    }
}

function setEditModelLoadingState(loading) {
    isEditModelLoading = loading;
    
    const loadButtons = [
        document.getElementById('btnEditLoadModel'),
        document.getElementById('btnLoadEditModelSettings')
    ];
    const unloadButtons = [
        document.getElementById('btnEditUnloadModel'),
        document.getElementById('btnUnloadEditModelSettings')
    ];
    
    loadButtons.forEach(btn => {
        if (!btn) return;
        btn.disabled = loading;
        
        // 버튼별 원래 라벨 유지
        if (loading) {
            btn.innerHTML = '<i class="ri-loader-4-line"></i> 로딩...';
        } else {
            if (btn.id === 'btnLoadEditModelSettings') {
                btn.innerHTML = '<i class="ri-download-line"></i> 편집 모델 로드';
            } else {
                btn.innerHTML = '<i class="ri-download-line"></i> 로드';
            }
        }
    });
    
    unloadButtons.forEach(btn => {
        if (btn) btn.disabled = loading;
    });
    
    const statusBadge = document.getElementById('editModelStatusBadge');
    if (statusBadge && loading) {
        const dot = statusBadge.querySelector('.status-dot');
        const text = statusBadge.querySelector('.status-text');
        if (dot) dot.classList.add('loading');
        if (text) text.textContent = '로딩 중...';
    }
}

async function updateEditModelStatus() {
    try {
        const status = await apiCall('/edit/status');
        updateEditModelStatusFromData(status);
    } catch (error) {
        console.error('편집 모델 상태 업데이트 실패:', error);
    }
}

function updateEditModelStatusFromData(data) {
    const statusBadge = document.getElementById('editModelStatusBadge');
    if (!statusBadge) return;
    
    const dot = statusBadge.querySelector('.status-dot');
    const text = statusBadge.querySelector('.status-text');
    
    if (data.model_loaded) {
        if (dot) {
            dot.classList.remove('offline', 'loading');
            dot.classList.add('online');
        }
        if (text) {
            if (data.current_model) {
                text.textContent = `✓ ${data.current_model}`;
                statusBadge.title = data.current_model;
            } else {
                text.textContent = '편집 모델 로드됨';
                statusBadge.title = '';
            }
        }
    } else {
        if (dot) {
            dot.classList.remove('online', 'loading');
            dot.classList.add('offline');
        }
        if (text) {
            text.textContent = '편집 모델 미로드';
        }
        statusBadge.title = '';
    }
}


// ============= 편집 실행 =============
async function executeEdit() {
    if (isEditing) {
        addEditMessage('system', '⚠️ 이미 편집 중입니다.');
        return;
    }
    
    if (!editImageFile) {
        addEditMessage('system', '❌ 편집할 이미지를 업로드해주세요.');
        return;
    }
    
    const koreanText = document.getElementById('editKoreanInput')?.value?.trim() || '';
    let prompt = document.getElementById('editPromptInput')?.value?.trim() || '';
    
    // 한국어가 있고 영어가 없으면 번역
    if (koreanText && !prompt) {
        addEditMessage('system', '🌐 번역 후 편집합니다...');
        const translated = await translateEditKoreanInput();
        if (!translated) {
            addEditMessage('system', '❌ 번역 실패로 편집을 중단합니다.');
            return;
        }
        prompt = document.getElementById('editPromptInput')?.value?.trim() || '';
    }
    
    if (!prompt) {
        addEditMessage('system', '❌ 편집 프롬프트를 입력해주세요.');
        return;
    }
    
    isEditing = true;
    setEditButtonState(true);
    
    // 사용자 메시지 표시
    const displayPrompt = koreanText ? `🇰🇷 ${koreanText}\n🇺🇸 ${prompt}` : prompt;
    addEditMessage('user', displayPrompt);
    
    // 진행률 표시 시작
    showEditProgress('편집 준비 중...', 0);
    
    const formData = new FormData();
    formData.append('image', editImageFile);
    formData.append('prompt', prompt);
    formData.append('korean_prompt', koreanText);
    formData.append('steps', document.getElementById('editStepsInput')?.value || '50');
    formData.append('guidance_scale', document.getElementById('editGuidanceInput')?.value || '4.5');
    formData.append('seed', document.getElementById('editSeedInput')?.value || '-1');
    formData.append('num_images', document.getElementById('editNumImagesInput')?.value || '1');
    formData.append('auto_translate', 'false');  // 이미 번역했으므로
    
    if (referenceImageFile) {
        formData.append('reference_image', referenceImageFile);
    }
    
    try {
        const response = await fetch('/api/edit/generate', {
            method: 'POST',
            body: formData,
            credentials: 'include'
        });
        
        if (!response.ok) {
            const error = await response.json();
            // detail이 객체인 경우 (ValidationError 등) 처리
            let errorMessage = '편집 실패';
            if (error.detail) {
                if (typeof error.detail === 'string') {
                    errorMessage = error.detail;
                } else if (Array.isArray(error.detail)) {
                    // FastAPI ValidationError 형식
                    errorMessage = error.detail.map(e => e.msg || e.message || JSON.stringify(e)).join(', ');
                } else if (typeof error.detail === 'object') {
                    errorMessage = JSON.stringify(error.detail);
                }
            }
            throw new Error(errorMessage);
        }
        
        // 결과는 WebSocket으로 받음
        
    } catch (error) {
        addEditMessage('system', `❌ 오류: ${error.message}`, 'error');
        hideEditProgress();
        isEditing = false;
        setEditButtonState(false);
    }
}

function setEditButtonState(editing) {
    const btnEdit = document.getElementById('btnEdit');
    if (btnEdit) {
        btnEdit.disabled = editing;
        btnEdit.innerHTML = editing ? '<i class="ri-loader-4-line"></i> 편집 중...' : '<i class="ri-edit-line"></i> 편집';
    }
}


// ============= 편집 LLM 기능 =============
let isEditLlmProcessing = false;  // 편집 탭 LLM 처리 중 여부
const EDIT_LLM_TIMEOUT = 5000;    // 번역, 향상 타임아웃 (5초)
const EDIT_SUGGEST_TIMEOUT = 10000;  // 편집제안 타임아웃 (10초)

function setEditLlmButtonsDisabled(disabled) {
    const buttons = [
        document.getElementById('btnEditTranslate'),
        document.getElementById('btnEditEnhance'),
        document.getElementById('btnEditSuggest'),
        document.getElementById('btnEditTranslateKorean')
    ];
    
    buttons.forEach(btn => {
        if (btn) {
            btn.disabled = disabled;
            btn.style.opacity = disabled ? '0.5' : '1';
            btn.style.pointerEvents = disabled ? 'none' : 'auto';
        }
    });
    
    isEditLlmProcessing = disabled;
}

async function editApiCallWithTimeout(endpoint, method, body, timeout) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        signal: controller.signal
    };
    
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    try {
        const response = await fetch(`/api${endpoint}`, options);
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '요청 실패');
        }
        
        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error(`요청 시간 초과 (${timeout / 1000}초)`);
        }
        throw error;
    }
}

async function translateEditKoreanInput() {
    const koreanInput = document.getElementById('editKoreanInput');
    const koreanText = koreanInput?.value?.trim();
    const statusEl = document.getElementById('editTranslateStatus');
    
    if (!koreanText) {
        addEditMessage('system', '⚠️ 한국어 입력창에 텍스트를 입력해주세요.');
        return false;
    }
    
    if (isEditLlmProcessing) {
        return false;
    }
    
    try {
        setEditLlmButtonsDisabled(true);
        if (statusEl) {
            statusEl.textContent = '번역 중...';
            statusEl.className = 'translate-status translating';
        }
        
        const result = await editApiCallWithTimeout('/edit/translate', 'POST', { text: koreanText }, EDIT_LLM_TIMEOUT);
        
        if (result.success) {
            document.getElementById('editPromptInput').value = result.translated;
            if (statusEl) {
                statusEl.textContent = '✓ 번역됨';
                statusEl.className = 'translate-status success';
                setTimeout(() => {
                    statusEl.textContent = '';
                    statusEl.className = 'translate-status';
                }, 2000);
            }
            return true;
        }
        return false;
    } catch (error) {
        if (statusEl) {
            statusEl.textContent = '번역 실패';
            statusEl.className = 'translate-status error';
        }
        addEditMessage('system', `❌ 번역 실패: ${error.message}`, 'error');
        return false;
    } finally {
        setEditLlmButtonsDisabled(false);
    }
}

async function translateEditPrompt() {
    const koreanInput = document.getElementById('editKoreanInput');
    const text = koreanInput?.value?.trim() || document.getElementById('editPromptInput')?.value?.trim();
    if (!text) return;
    
    if (isEditLlmProcessing) return;
    
    try {
        setEditLlmButtonsDisabled(true);
        addEditMessage('system', '🌐 번역 중...');
        const result = await editApiCallWithTimeout('/edit/translate', 'POST', { text }, EDIT_LLM_TIMEOUT);
        
        if (result.success) {
            document.getElementById('editPromptInput').value = result.translated;
            addEditMessage('system', '✅ 번역 완료');
        }
    } catch (error) {
        addEditMessage('system', `❌ 번역 실패: ${error.message}`, 'error');
    } finally {
        setEditLlmButtonsDisabled(false);
    }
}

async function enhanceEditPrompt() {
    const prompt = document.getElementById('editPromptInput')?.value?.trim();
    if (!prompt) {
        addEditMessage('system', '⚠️ 영어 프롬프트를 먼저 입력해주세요.');
        return;
    }
    
    if (isEditLlmProcessing) return;
    
    try {
        setEditLlmButtonsDisabled(true);
        addEditMessage('system', '✨ 편집 지시어 향상 중...');
        const result = await editApiCallWithTimeout('/edit/enhance', 'POST', { instruction: prompt }, EDIT_LLM_TIMEOUT);
        
        if (result.success) {
            document.getElementById('editPromptInput').value = result.enhanced;
            addEditMessage('system', '✅ 편집 지시어 향상 완료');
        }
    } catch (error) {
        addEditMessage('system', `❌ 향상 실패: ${error.message}`, 'error');
    } finally {
        setEditLlmButtonsDisabled(false);
    }
}

async function suggestEdits() {
    if (isEditLlmProcessing) return;
    
    try {
        setEditLlmButtonsDisabled(true);
        addEditMessage('system', '💡 편집 아이디어 생성 중...');
        const result = await editApiCallWithTimeout('/edit/suggest', 'POST', { context: '', image_description: '' }, EDIT_SUGGEST_TIMEOUT);
        
        if (result.success && result.suggestions_korean.length > 0) {
            let html = '<p>💡 <strong>편집 아이디어:</strong></p><ul>';
            result.suggestions_korean.forEach((suggestion, i) => {
                html += `<li style="cursor:pointer;" onclick="applyEditSuggestion('${escapeHtml(result.suggestions[i])}', '${escapeHtml(suggestion)}')">${suggestion}</li>`;
            });
            html += '</ul>';
            addEditMessage('system', html);
        }
    } catch (error) {
        addEditMessage('system', `❌ 제안 생성 실패: ${error.message}`, 'error');
    } finally {
        setEditLlmButtonsDisabled(false);
    }
}

function applyEditSuggestion(english, korean) {
    document.getElementById('editKoreanInput').value = korean;
    document.getElementById('editPromptInput').value = english;
    addEditMessage('system', '✅ 편집 제안이 적용되었습니다.');
}


// ============= 편집 메시지 표시 =============
function addEditMessage(type, content, style = '') {
    const messagesEl = document.getElementById('editMessages');
    if (!messagesEl) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type} ${style}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<p>${content}</p>`;
    
    messageDiv.appendChild(contentDiv);
    messagesEl.appendChild(messageDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addEditImageMessage(originalSrc, resultImages, prompt) {
    const messagesEl = document.getElementById('editMessages');
    if (!messagesEl) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant edit-result';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // 이미지 목록 생성 (원본 + 결과들)
    const imageList = [
        { path: originalSrc, metadata: { prompt: '원본 이미지' } },
        ...resultImages.map(img => ({
            path: img.base64 ? 'data:image/png;base64,' + img.base64 : img.path,
            metadata: { prompt: `편집 결과: ${prompt}`, seed: img.seed }
        }))
    ];
    
    // 원본 → 결과 비교
    const comparisonDiv = document.createElement('div');
    comparisonDiv.className = 'edit-comparison';
    
    // 원본 이미지
    const originalWrapper = document.createElement('div');
    originalWrapper.className = 'edit-result-image-wrapper';

    const originalImg = document.createElement('img');
    originalImg.src = originalSrc;
    originalImg.alt = '원본';
    originalImg.title = '원본 이미지 (클릭하여 확대)';
    originalImg.onclick = () => showImageModalWithList(imageList, 0);

    const originalContinueBtn = document.createElement('button');
    originalContinueBtn.type = 'button';
    originalContinueBtn.className = 'continue-edit-btn';
    originalContinueBtn.title = '이 이미지를 입력 이미지로 넣고 이어서 편집';
    originalContinueBtn.innerHTML = '<i class="ri-add-line"></i> 이어서 편집';
    originalContinueBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        continueEditFromMessageImage(originalImg.src);
    });

    originalWrapper.appendChild(originalImg);
    originalWrapper.appendChild(originalContinueBtn);
    comparisonDiv.appendChild(originalWrapper);
    
    // 화살표
    const arrow = document.createElement('span');
    arrow.className = 'edit-arrow';
    arrow.innerHTML = '<i class="ri-arrow-right-line"></i>';
    comparisonDiv.appendChild(arrow);
    
    // 결과 이미지들
    resultImages.forEach((img, index) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'edit-result-image-wrapper';

        const resultImg = document.createElement('img');
        resultImg.src = img.base64 ? 'data:image/png;base64,' + img.base64 : img.path;
        resultImg.alt = '결과';
        resultImg.title = `시드: ${img.seed}\n클릭하여 확대 (좌우 화살표로 탐색)`;
        resultImg.onclick = () => showImageModalWithList(imageList, index + 1);

        const continueBtn = document.createElement('button');
        continueBtn.type = 'button';
        continueBtn.className = 'continue-edit-btn';
        continueBtn.title = '이 이미지를 입력 이미지로 넣고 이어서 편집';
        continueBtn.innerHTML = '<i class="ri-add-line"></i> 이어서 편집';
        continueBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            continueEditFromMessageImage(resultImg.src);
        });

        wrapper.appendChild(resultImg);
        wrapper.appendChild(continueBtn);
        comparisonDiv.appendChild(wrapper);
    });
    
    contentDiv.appendChild(comparisonDiv);
    messageDiv.appendChild(contentDiv);
    messagesEl.appendChild(messageDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}


// ============= 편집 프로그레스 =============
function showEditProgress(label, percent) {
    const container = document.getElementById('editProgressContainer');
    const labelEl = document.getElementById('editProgressLabel');
    const percentEl = document.getElementById('editProgressPercent');
    const fillEl = document.getElementById('editProgressFill');
    
    if (container) container.style.display = 'block';
    if (labelEl) labelEl.textContent = label;
    if (percentEl) percentEl.textContent = `${Math.round(percent)}%`;
    if (fillEl) fillEl.style.width = `${percent}%`;
}

function updateEditProgress(percent, label, detail) {
    const labelEl = document.getElementById('editProgressLabel');
    const percentEl = document.getElementById('editProgressPercent');
    const fillEl = document.getElementById('editProgressFill');
    const detailEl = document.getElementById('editProgressDetail');
    
    if (label && labelEl) labelEl.textContent = label;
    if (percentEl) percentEl.textContent = `${Math.round(percent)}%`;
    if (fillEl) fillEl.style.width = `${percent}%`;
    if (detail && detailEl) detailEl.textContent = detail;
}

function hideEditProgress() {
    const container = document.getElementById('editProgressContainer');
    const detailEl = document.getElementById('editProgressDetail');
    
    if (container) container.style.display = 'none';
    if (detailEl) detailEl.textContent = '';
}


// ============= 편집 양자화 옵션 로드 =============
async function loadEditQuantizationOptions() {
    try {
        const result = await apiCall('/edit/status');
        const settingsSelect = document.getElementById('editQuantizationSelectSettings');
        
        if (result.quantization_options && settingsSelect) {
            settingsSelect.innerHTML = '';
            result.quantization_options.forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option;
                settingsSelect.appendChild(opt);
            });
        }

        // /settings에서 먼저 내려온 편집 양자화 설정값이 있으면 반영
        if (settingsSelect && pendingEditQuantizationValue && Array.from(settingsSelect.options).some(o => o.value === pendingEditQuantizationValue)) {
            settingsSelect.value = pendingEditQuantizationValue;
            pendingEditQuantizationValue = null;
        }
        
        updateEditModelStatusFromData(result);
    } catch (error) {
        console.error('편집 양자화 옵션 로드 실패:', error);
    }
}


// ============= 편집 히스토리 =============
async function loadEditHistory() {
    try {
        const result = await apiCall('/edit/history');
        const list = document.getElementById('editHistoryList');
        if (!list) return;
        
        list.innerHTML = '';
        
        result.history.forEach(entry => {
            const item = document.createElement('div');
            item.className = 'edit-history-item';
            
            // 이미지에 클릭 이벤트를 위해 데이터 저장
            const originalPath = entry.original_image_path || '';
            const resultPath = (entry.result_image_paths && entry.result_image_paths.length > 0) ? entry.result_image_paths[0] : '';
            
            let imagesHtml = '<div class="edit-history-images">';
            if (originalPath) {
                imagesHtml += `<div class="edit-history-image-wrapper"><img src="${originalPath}" alt="원본" data-path="${originalPath}" data-type="original"></div>`;
            }
            imagesHtml += '<span class="edit-history-arrow"><i class="ri-arrow-right-line"></i></span>';
            if (resultPath) {
                imagesHtml += `<div class="edit-history-image-wrapper"><img src="${resultPath}" alt="결과" data-path="${resultPath}" data-type="result"></div>`;
            }
            imagesHtml += '</div>';
            
            const hasKorean = entry.korean_prompt && entry.korean_prompt.trim();
            const chainBadge = entry.parent_id ? '<div class="edit-history-chain-badge"><i class="ri-links-line"></i> 연속 편집</div>' : '';
            
            item.innerHTML = `
                <div class="edit-history-item-header">
                    <span class="edit-history-item-time">${formatDate(entry.timestamp)}</span>
                    <div class="item-actions">
                        <button class="btn btn-secondary" onclick="useEditHistory('${entry.id}')">사용</button>
                        <button class="btn btn-primary" onclick="continueEditHistory('${entry.id}')" title="이 결과 이미지로 추가 편집">
                            <i class="ri-add-line"></i> 이어서 편집
                        </button>
                    </div>
                </div>
                ${imagesHtml}
                ${hasKorean ? `<div class="edit-history-item-prompt"><span class="lang-badge kr">🇰🇷</span> ${escapeHtml(entry.korean_prompt)}</div>` : ''}
                <div class="edit-history-item-prompt"><span class="lang-badge us">🇺🇸</span> ${escapeHtml(entry.prompt)}</div>
                ${chainBadge}
            `;
            
            // 이미지 목록 생성 (네비게이션용)
            const historyImageList = [];
            if (originalPath) {
                historyImageList.push({
                    path: originalPath,
                    metadata: { prompt: `원본 이미지\n편집 프롬프트: ${entry.prompt}` }
                });
            }
            if (resultPath) {
                historyImageList.push({
                    path: resultPath,
                    metadata: { prompt: `편집 결과\n편집 프롬프트: ${entry.prompt}`, seed: entry.seed }
                });
            }
            
            // 이미지 클릭 이벤트 추가
            item.querySelectorAll('.edit-history-image-wrapper img').forEach((img, imgIndex) => {
                img.style.cursor = 'pointer';
                img.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const clickedIndex = img.dataset.type === 'original' ? 0 : (originalPath ? 1 : 0);
                    showImageModalWithList(historyImageList, clickedIndex);
                });
            });
            
            list.appendChild(item);
        });
    } catch (error) {
        console.error('편집 히스토리 로드 실패:', error);
    }
}

async function useEditHistory(historyId) {
    try {
        const result = await apiCall(`/edit/history/${historyId}`);
        const entry = result.history;
        
        document.getElementById('editPromptInput').value = entry.prompt;
        
        const koreanInput = document.getElementById('editKoreanInput');
        if (koreanInput) {
            koreanInput.value = entry.korean_prompt || '';
        }
        
        // 설정 복원
        if (entry.settings) {
            if (entry.settings.steps) document.getElementById('editStepsInput').value = entry.settings.steps;
            if (entry.settings.guidance_scale) document.getElementById('editGuidanceInput').value = entry.settings.guidance_scale;
            if (entry.settings.seed) document.getElementById('editSeedInput').value = entry.settings.seed;
        }
        
        switchTab('edit');
        addEditMessage('system', '✅ 편집 설정이 복원되었습니다. 이미지를 업로드하세요.');
    } catch (error) {
        addEditMessage('system', `❌ 히스토리 로드 실패: ${error.message}`, 'error');
    }
}

async function continueEditHistory(historyId) {
    try {
        const result = await apiCall(`/edit/history/${historyId}`);
        const entry = result.history;
        
        // 결과 이미지를 새 편집의 입력으로 사용
        if (entry.result_image_paths && entry.result_image_paths.length > 0) {
            const imagePath = entry.result_image_paths[0];
            
            // 이미지 로드하여 File 객체 생성
            const response = await fetch(imagePath);
            const blob = await response.blob();
            const file = new File([blob], 'continue_edit.png', { type: 'image/png' });
            
            handleEditImageUpload(file);
        }
        
        // 프롬프트 초기화
        document.getElementById('editKoreanInput').value = '';
        document.getElementById('editPromptInput').value = '';
        
        switchTab('edit');
        addEditMessage('system', '✅ 이전 편집 결과 이미지가 로드되었습니다. 새 편집 지시어를 입력하세요.');
    } catch (error) {
        addEditMessage('system', `❌ 히스토리 로드 실패: ${error.message}`, 'error');
    }
}

async function clearEditHistory() {
    if (!confirm('모든 편집 히스토리를 삭제하시겠습니까?')) return;
    
    try {
        await apiCall('/edit/history', 'DELETE');
        loadEditHistory();
        addEditMessage('system', '✅ 편집 히스토리 삭제됨');
    } catch (error) {
        addEditMessage('system', `❌ 삭제 실패: ${error.message}`, 'error');
    }
}


// ============= 인증 관련 함수 =============

/**
 * 현재 사용자 정보 로드
 */
async function loadCurrentUser() {
    try {
        const data = await apiCall('/auth/me', 'GET');
        
        // 관리자 배지 표시
        const adminBadge = document.getElementById('adminBadge');
        if (adminBadge) {
            adminBadge.style.display = data.is_admin ? 'inline-block' : 'none';
        }
        
        // 관리자 전용 섹션 표시
        if (data.is_admin) {
            const userManagementSection = document.getElementById('userManagementSection');
            if (userManagementSection) {
                userManagementSection.style.display = 'block';
                loadUserList();  // 사용자 목록 로드
            }
            
            const sessionManagementSection = document.getElementById('sessionManagementSection');
            if (sessionManagementSection) {
                sessionManagementSection.style.display = 'block';
                loadSessionList();  // 세션 목록 로드
            }
        }
        
        return data;
    } catch (error) {
        console.error('Failed to load current user:', error);
        // 인증 실패 시 로그인 페이지로 리다이렉트
        if (error.status === 401) {
            window.location.href = '/login';
        }
        return null;
    }
}

/**
 * 로그아웃
 */
async function logout() {
    if (!confirm('로그아웃 하시겠습니까?')) return;
    
    try {
        await apiCall('/auth/logout', 'POST');
        window.location.href = '/login';
    } catch (error) {
        console.error('Logout error:', error);
        // 에러가 나도 로그인 페이지로 이동
        window.location.href = '/login';
    }
}

/**
 * 비밀번호 변경 모달 열기
 */
function openChangePasswordModal() {
    const modal = document.getElementById('changePasswordModal');
    if (modal) {
        modal.classList.add('active');
        // 폼 초기화
        document.getElementById('currentPasswordInput').value = '';
        document.getElementById('newPasswordInput').value = '';
        document.getElementById('newPasswordConfirmInput').value = '';
        const errorDiv = document.getElementById('changePasswordError');
        if (errorDiv) {
            errorDiv.textContent = '';
            errorDiv.style.display = 'none';
        }
    }
}

/**
 * 비밀번호 변경 처리
 */
async function handleChangePassword(e) {
    e.preventDefault();
    
    const currentPassword = document.getElementById('currentPasswordInput').value;
    const newPassword = document.getElementById('newPasswordInput').value;
    const newPasswordConfirm = document.getElementById('newPasswordConfirmInput').value;
    const errorDiv = document.getElementById('changePasswordError');
    
    // 클라이언트 검증
    if (!currentPassword || !newPassword || !newPasswordConfirm) {
        showFormError(errorDiv, '모든 항목을 입력해주세요.');
        return;
    }
    
    if (newPassword !== newPasswordConfirm) {
        showFormError(errorDiv, '새 비밀번호가 일치하지 않습니다.');
        return;
    }
    
    if (newPassword.length < 4) {
        showFormError(errorDiv, '비밀번호는 4자 이상이어야 합니다.');
        return;
    }
    
    try {
        await apiCall('/auth/change-password', 'POST', {
            current_password: currentPassword,
            new_password: newPassword,
            new_password_confirm: newPasswordConfirm
        });
        
        closeModal('changePasswordModal');
        addMessage('system', '✅ 비밀번호가 변경되었습니다.');
    } catch (error) {
        showFormError(errorDiv, error.message || '비밀번호 변경에 실패했습니다.');
    }
}

/**
 * 폼 에러 표시
 */
function showFormError(element, message) {
    if (element) {
        element.textContent = message;
        element.style.display = 'block';
    }
}

/**
 * 관리자: 사용자 목록 로드
 */
async function loadUserList() {
    try {
        const data = await apiCall('/admin/users', 'GET');
        renderUserList(data.users || []);
    } catch (error) {
        console.error('Failed to load user list:', error);
    }
}

/**
 * 관리자: 사용자 목록 렌더링
 */
function renderUserList(users) {
    const container = document.getElementById('userList');
    if (!container) return;
    
    if (users.length === 0) {
        container.innerHTML = '<div class="empty-message">등록된 사용자가 없습니다.</div>';
        return;
    }
    
    container.innerHTML = users.map(user => `
        <div class="user-item" data-user-id="${user.id}">
            <div class="user-item-info">
                <span class="user-item-name">${user.username}</span>
                <span class="user-item-date">가입: ${formatDate(user.created_at)}</span>
                <span class="user-item-login">${user.last_login ? '최근 로그인: ' + formatDate(user.last_login) : '로그인 기록 없음'}</span>
            </div>
            <div class="user-item-actions">
                <button class="btn btn-sm btn-secondary" onclick="resetUserPassword(${user.id}, '${user.username}')">
                    <i class="ri-lock-password-line"></i> 비밀번호 초기화
                </button>
                <button class="btn btn-sm btn-danger" onclick="deleteUser(${user.id}, '${user.username}')">
                    <i class="ri-delete-bin-line"></i> 삭제
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * 관리자: 사용자 비밀번호 초기화
 */
async function resetUserPassword(userId, username) {
    if (!confirm(`'${username}' 사용자의 비밀번호를 초기화하시겠습니까?`)) return;
    
    try {
        const data = await apiCall(`/admin/users/${userId}/reset-password`, 'POST', {});
        alert(`비밀번호가 초기화되었습니다.\n\n새 임시 비밀번호: ${data.new_password}\n\n이 비밀번호를 사용자에게 전달해주세요.`);
    } catch (error) {
        alert('비밀번호 초기화 실패: ' + (error.message || '알 수 없는 오류'));
    }
}

/**
 * 관리자: 사용자 삭제
 */
async function deleteUser(userId, username) {
    if (!confirm(`'${username}' 사용자를 삭제하시겠습니까?\n\n모든 데이터가 삭제됩니다.`)) return;
    
    try {
        await apiCall(`/admin/users/${userId}`, 'DELETE');
        loadUserList();
        addMessage('system', `✅ '${username}' 사용자가 삭제되었습니다.`);
    } catch (error) {
        alert('사용자 삭제 실패: ' + (error.message || '알 수 없는 오류'));
    }
}

/**
 * 날짜 포맷팅
 */
function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString('ko-KR');
}