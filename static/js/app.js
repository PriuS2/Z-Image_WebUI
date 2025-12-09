// Z-Image WebUI - JavaScript

// ============= 전역 변수 =============
let ws = null;
let isGenerating = false;
let isModelLoading = false;
let templates = {};
let isTranslating = false;
let lastHistoryId = null;  // 마지막으로 저장된 히스토리 ID

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
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onclose = () => {
        console.log('WebSocket 연결 끊김, 재연결 시도...');
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket 오류:', error);
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'connected':
            addMessage('system', data.content);
            break;
        case 'system':
        case 'warning':
            addMessage('system', data.content);
            // 모델 로딩 메시지 분석하여 프로그레스 업데이트
            updateProgressFromMessage(data.content);
            break;
        case 'progress':
            addMessage('system', data.content);
            updateProgressFromMessage(data.content);
            break;
        case 'model_progress':
            // 모델 다운로드/로드 프로그레스 전용
            updateModelProgress(data.progress, data.label, data.detail, data.stage || '');
            setModelLoadingState(data.stage !== 'complete' && data.stage !== 'error');
            break;
        case 'complete':
            addMessage('system', data.content);
            updateModelStatus();
            hideProgress();
            break;
        case 'error':
            addMessage('system', data.content, 'error');
            hideProgress();
            break;
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
    
    // 단계에 따른 스타일 변경
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
    // 메시지에서 프로그레스 추정
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
        // 이미지 생성 진행률 파싱 (예: 1/4)
        const match = message.match(/\((\d+)\/(\d+)\)/);
        if (match) {
            const current = parseInt(match[1]);
            const total = parseInt(match[2]);
            const percent = (current / total) * 100;
            showProgress(`이미지 생성 중... (${current}/${total})`, percent);
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
}

function addImageMessage(images, prompt) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const imagesDiv = document.createElement('div');
    imagesDiv.className = 'message-images';
    
    images.forEach(img => {
        const imgEl = document.createElement('img');
        imgEl.src = `data:image/png;base64,${img.base64}`;
        imgEl.alt = prompt;
        imgEl.title = `시드: ${img.seed}\n클릭하여 확대`;
        imgEl.dataset.path = img.path;  // 이미지 경로 저장 (복원용)
        imgEl.onclick = () => showImageModal(img.path, img);
        imagesDiv.appendChild(imgEl);
    });
    
    contentDiv.appendChild(imagesDiv);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============= 대화 내용 관리 =============
// 현재 대화 내용을 JSON 형태로 추출
function getConversation() {
    const messages = [];
    const messageElements = chatMessages.querySelectorAll('.message');
    
    messageElements.forEach(msgEl => {
        const type = msgEl.classList.contains('user') ? 'user' :
                     msgEl.classList.contains('assistant') ? 'assistant' : 'system';
        
        const contentEl = msgEl.querySelector('.message-content');
        if (!contentEl) return;
        
        // 텍스트 메시지
        const textEl = contentEl.querySelector('p');
        const text = textEl ? textEl.innerHTML : '';
        
        // 이미지 메시지
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

// 대화 내용 복원
function restoreConversation(conversation) {
    // 기존 대화 내용 삭제 (환영 메시지 제외)
    const existingMessages = chatMessages.querySelectorAll('.message');
    existingMessages.forEach(msg => msg.remove());
    
    // 대화 내용 복원
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
            
            msg.images.forEach(imgData => {
                const imgEl = document.createElement('img');
                // 경로 처리 - base64이면 그대로, 상대경로면 그대로
                imgEl.src = imgData.path;
                imgEl.alt = imgData.alt || '';
                imgEl.dataset.path = imgData.path;
                imgEl.onclick = () => showImageModal(imgData.path, { prompt: imgData.alt });
                imagesDiv.appendChild(imgEl);
            });
            
            contentDiv.appendChild(imagesDiv);
        }
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
    });
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 히스토리에 대화 내용 저장
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
        headers: { 'Content-Type': 'application/json' }
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

// ============= 이미지 생성 =============
async function generateImage(preview = false) {
    if (isGenerating) {
        alert('이미 생성 중입니다.');
        return;
    }
    
    const koreanText = document.getElementById('koreanInput')?.value?.trim() || '';
    let prompt = promptInput.value.trim();
    
    // 한국어 입력이 있고 영어 프롬프트가 비어있으면 먼저 번역
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
    
    // 사용자 메시지 표시 (한국어가 있으면 둘 다 표시)
    if (koreanText && koreanText !== prompt) {
        addMessage('user', `🇰🇷 ${koreanText}\n🇺🇸 ${prompt}`);
    } else {
        addMessage('user', prompt);
    }
    
    // 해상도 처리 - 커스텀 또는 프리셋
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
        korean_prompt: koreanText,  // 한국어 프롬프트도 함께 전송
        width,
        height,
        steps: parseInt(document.getElementById('stepsInput').value) || 8,
        seed: parseInt(document.getElementById('seedInput').value) || -1,
        num_images: preview ? 1 : parseInt(document.getElementById('numImagesInput').value) || 1,
        auto_translate: false  // UI에서 이미 번역됨
    };
    
    try {
        const endpoint = preview ? '/preview' : '/generate';
        const result = await apiCall(endpoint, 'POST', requestBody);
        
        if (result.success && result.images) {
            addImageMessage(result.images, result.prompt);
            
            // 히스토리에 대화 내용 저장
            if (result.history_id) {
                lastHistoryId = result.history_id;
                // 약간의 딜레이 후 대화 내용 저장 (이미지가 DOM에 추가된 후)
                setTimeout(() => {
                    saveConversationToHistory(result.history_id);
                }, 500);
            }
        }
    } catch (error) {
        addMessage('system', `❌ 오류: ${error.message}`, 'error');
    } finally {
        isGenerating = false;
        setGenerateButtonState(false);
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
    
    // 양자화 옵션을 드롭다운에서 가져오기 (대화탭 또는 설정탭에서)
    const quantization = fromChat
        ? document.getElementById('chatQuantizationSelect')?.value || "BF16 (기본, 최고품질)"
        : document.getElementById('quantizationSelect')?.value || "BF16 (기본, 최고품질)";
    const modelPath = document.getElementById('modelPathInput')?.value || '';
    
    // CPU 오프로딩 체크 (대화탭 또는 설정탭에서)
    const cpuOffload = fromChat 
        ? document.getElementById('chatCpuOffloadCheck')?.checked || false
        : document.getElementById('cpuOffloadCheck')?.checked || false;
    
    try {
        setModelLoadingState(true);
        const offloadMsg = cpuOffload ? ' (CPU 오프로딩 사용)' : '';
        addMessage('system', `🔄 모델 로딩을 시작합니다...${offloadMsg}`);
        showProgress('모델 로딩 준비 중...', 5);
        
        await apiCall('/model/load', 'POST', {
            quantization,
            model_path: modelPath,
            cpu_offload: cpuOffload
        });
        
        updateModelStatus();
        updateProgress(100, '모델 로드 완료!');
        setTimeout(hideProgress, 1500);
        
        // 다운로드 상태 업데이트 (새로 다운로드된 모델 반영)
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
        setModelLoadingState(true);  // 언로드 시작 시 버튼 비활성화
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
        setModelLoadingState(false);  // 완료 후 버튼 다시 활성화
    }
}

function setModelLoadingState(loading) {
    isModelLoading = loading;
    
    // 버튼 상태 업데이트
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
    
    // 모델 상태 배지 업데이트
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
        
        // 사이드바 상태
        const indicator = modelStatus.querySelector('.status-indicator');
        const text = modelStatus.querySelector('span');
        
        // 대화탭 상태 배지
        const statusBadge = document.getElementById('modelStatusBadge');
        const dot = statusBadge?.querySelector('.status-dot');
        const badgeText = statusBadge?.querySelector('.status-text');
        
        if (status.model_loaded) {
            // 사이드바
            indicator.classList.add('online');
            indicator.classList.remove('offline');
            text.textContent = '모델 로드됨';
            
            // 대화탭
            if (dot) {
                dot.classList.remove('offline', 'loading');
                dot.classList.add('online');
            }
            if (badgeText) {
                badgeText.textContent = status.current_model ? `✓ ${status.current_model.split(' ')[0]}` : '모델 로드됨';
            }
        } else {
            // 사이드바
            indicator.classList.remove('online');
            indicator.classList.add('offline');
            text.textContent = '모델 미로드';
            
            // 대화탭
            if (dot) {
                dot.classList.remove('online', 'loading');
                dot.classList.add('offline');
            }
            if (badgeText) badgeText.textContent = '모델 미로드';
        }
    } catch (error) {
        console.error('상태 업데이트 실패:', error);
    }
}

// ============= 프롬프트 도구 =============

// 한국어 텍스트 감지
function isKorean(text) {
    const koreanRegex = /[가-힣]/;
    return koreanRegex.test(text);
}

// 한국어 입력창 번역 (버튼 클릭)
async function translateKoreanInput() {
    const koreanInputEl = document.getElementById('koreanInput');
    const koreanText = koreanInputEl?.value?.trim();
    const statusEl = document.getElementById('translateStatus');
    
    if (!koreanText) {
        addMessage('system', '⚠️ 한국어 입력창에 텍스트를 입력해주세요.');
        return false;
    }
    
    // 이미 번역 중이면 대기
    if (isTranslating) {
        return false;
    }
    
    // 한국어가 포함되어 있지 않으면 그대로 복사
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
        if (statusEl) {
            statusEl.textContent = '번역 중...';
            statusEl.className = 'translate-status translating';
        }
        
        const result = await apiCall('/translate', 'POST', { text: koreanText });
        
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
    }
}

// 기존 번역 버튼 (옵션바의 번역 버튼)
async function translatePrompt() {
    const koreanInputEl = document.getElementById('koreanInput');
    const text = koreanInputEl?.value?.trim() || promptInput.value.trim();
    if (!text) return;
    
    try {
        addMessage('system', '🌐 번역 중...');
        const result = await apiCall('/translate', 'POST', { text });
        
        if (result.success) {
            promptInput.value = result.translated;
            addMessage('system', '✅ 번역 완료');
        }
    } catch (error) {
        addMessage('system', `❌ 번역 실패: ${error.message}`, 'error');
    }
}

// 번역이 필요한지 확인 (한국어 입력창에 텍스트가 있고, 영어 프롬프트가 비어있거나 다른 경우)
function needsTranslation() {
    const koreanText = document.getElementById('koreanInput')?.value?.trim() || '';
    const englishText = document.getElementById('promptInput')?.value?.trim() || '';
    
    // 한국어 입력이 있고, 영어가 비어있으면 번역 필요
    if (koreanText && !englishText) {
        return true;
    }
    
    // 한국어 입력이 있고, 한국어가 포함되어 있으면 번역 필요
    if (koreanText && isKorean(koreanText)) {
        return true;
    }
    
    return false;
}

async function enhancePrompt() {
    const prompt = promptInput.value.trim();
    if (!prompt) return;
    
    const koreanInputEl = document.getElementById('koreanInput');
    const statusEl = document.getElementById('translateStatus');
    
    try {
        addMessage('system', '✨ 프롬프트 향상 중...');
        const result = await apiCall('/enhance', 'POST', { prompt, style: '기본' });
        
        if (result.success) {
            promptInput.value = result.enhanced;
            addMessage('system', '✅ 프롬프트 향상 완료');
            
            // 향상된 영어 프롬프트를 한국어로 역번역
            if (koreanInputEl) {
                try {
                    if (statusEl) {
                        statusEl.textContent = '한국어 변환 중...';
                        statusEl.className = 'translate-status translating';
                    }
                    
                    const reverseResult = await apiCall('/translate-reverse', 'POST', { text: result.enhanced });
                    
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
                    // 역번역 실패해도 향상은 성공했으므로 에러 메시지 안 띄움
                    if (statusEl) {
                        statusEl.textContent = '';
                        statusEl.className = 'translate-status';
                    }
                }
            }
        }
    } catch (error) {
        addMessage('system', `❌ 향상 실패: ${error.message}`, 'error');
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
    
    // 변수 기본값 적용
    if (template.variables) {
        for (const [key, value] of Object.entries(template.variables)) {
            prompt = prompt.replace(`{${key}}`, value);
        }
    }
    
    promptInput.value = prompt;
    // 한국어 입력창 비우기 (영어 템플릿 직접 사용)
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
        const chatSelect = document.getElementById('chatQuantizationSelect');
        
        if (result.quantization_options) {
            // 설정 탭과 대화 탭 드롭다운 모두 채우기
            [settingsSelect, chatSelect].forEach(select => {
                if (select) {
                    select.innerHTML = '';
                    
                    result.quantization_options.forEach(option => {
                        const opt = document.createElement('option');
                        opt.value = option;
                        // 대화 탭에서는 짧게 표시
                        if (select === chatSelect) {
                            // "GGUF Q8_0 (7.22GB, 고품질)" -> "Q8_0 (7.22GB)"
                            // "BF16 (기본, 최고품질)" -> "BF16 (최고품질)"
                            let shortName = option;
                            const match = option.match(/^(?:GGUF\s+)?(\S+)\s*\(([^,]+)/);
                            if (match) {
                                shortName = `${match[1]} (${match[2].trim()})`;
                            }
                            opt.textContent = shortName;
                            opt.title = option;  // 전체 이름은 툴팁으로
                        } else {
                            opt.textContent = option;
                        }
                        select.appendChild(opt);
                    });
                }
            });
            
            console.log('양자화 옵션 로드 완료:', result.quantization_options.length + '개');
            
            // 다운로드 상태 확인 및 표시
            updateModelDownloadStatus();
        }
    } catch (error) {
        console.error('양자화 옵션 로드 실패:', error);
    }
}

// 모델 다운로드 상태 업데이트
async function updateModelDownloadStatus() {
    try {
        const result = await apiCall('/model-status');
        const status = result.status || {};
        
        const settingsSelect = document.getElementById('quantizationSelect');
        const chatSelect = document.getElementById('chatQuantizationSelect');
        
        [settingsSelect, chatSelect].forEach(select => {
            if (!select) return;
            
            Array.from(select.options).forEach(opt => {
                const optionName = opt.value;
                const isDownloaded = status[optionName] || false;
                
                // 기존 텍스트에서 다운로드 표시 제거
                let text = opt.textContent.replace(/^[✓⬇]\s*/, '');
                
                // 다운로드 상태에 따라 표시
                if (isDownloaded) {
                    opt.textContent = `✓ ${text}`;
                    opt.style.color = '#22c55e';  // 녹색
                } else {
                    opt.textContent = `⬇ ${text}`;
                    opt.style.color = '';  // 기본색
                }
                
                // 툴팁 업데이트
                const statusText = isDownloaded ? '(다운로드됨)' : '(미다운로드)';
                opt.title = `${optionName} ${statusText}`;
            });
        });
        
        console.log('모델 다운로드 상태 업데이트 완료');
    } catch (error) {
        console.error('모델 다운로드 상태 확인 실패:', error);
    }
}

// ============= 갤러리 =============
async function loadGallery() {
    try {
        const result = await apiCall('/gallery');
        const grid = document.getElementById('galleryGrid');
        grid.innerHTML = '';
        
        result.images.forEach(img => {
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
                <img src="${img.path}" alt="${img.filename}">
                <div class="gallery-item-overlay">
                    <span>${img.filename}</span>
                </div>
            `;
            item.onclick = () => showImageModal(img.path, img.metadata);
            grid.appendChild(item);
        });
    } catch (error) {
        console.error('갤러리 로드 실패:', error);
    }
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

// 히스토리 항목 사용 (한국어/영어 프롬프트 모두 복원)
async function useHistoryEntry(historyId) {
    try {
        const result = await apiCall(`/history/${historyId}`);
        const entry = result.history;
        
        // 영어 프롬프트 설정
        promptInput.value = entry.prompt;
        
        // 한국어 프롬프트 복원
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

// 레거시 호환 (이전 방식)
function useHistoryPrompt(prompt) {
    promptInput.value = prompt;
    const koreanInputEl = document.getElementById('koreanInput');
    if (koreanInputEl) koreanInputEl.value = '';
    switchTab('chat');
    addMessage('system', '✅ 프롬프트 적용됨');
}

// 히스토리에서 대화 내용 복원
async function restoreHistoryConversation(historyId) {
    try {
        const result = await apiCall(`/history/${historyId}`);
        const entry = result.history;
        
        if (entry.conversation && entry.conversation.length > 0) {
            // 확인 대화상자
            if (!confirm('현재 대화 내용을 지우고 히스토리의 대화를 복원하시겠습니까?')) {
                return;
            }
            
            // 프롬프트 설정 (영어)
            promptInput.value = entry.prompt;
            
            // 한국어 프롬프트 복원
            const koreanInputEl = document.getElementById('koreanInput');
            if (koreanInputEl) {
                koreanInputEl.value = entry.korean_prompt || '';
            }
            
            // 설정 복원
            if (entry.settings) {
                if (entry.settings.width && entry.settings.height) {
                    const resSelect = document.getElementById('resolutionSelect');
                    const resValue = `${entry.settings.width}x${entry.settings.height}`;
                    // 프리셋에 있으면 선택, 없으면 커스텀
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
            
            // 대화 내용 복원
            restoreConversation(entry.conversation);
            
            // 탭 전환
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
    // 한국어 입력창 비우기 (영어 프롬프트 직접 사용)
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

async function loadLlmProviders() {
    try {
        const result = await apiCall('/settings');
        llmProviders = result.llm_providers || {};
        
        const currentProvider = result.llm_provider || 'openai';
        const currentModel = result.llm_model || '';
        
        // 설정탭 프로바이더 셀렉트
        const providerSelect = document.getElementById('llmProviderSelect');
        // 대화탭 프로바이더 셀렉트
        const chatProviderSelect = document.getElementById('chatLlmProviderSelect');
        
        // 프로바이더 옵션 채우기
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
        
        // 모델 목록 업데이트 (설정탭 + 대화탭)
        updateLlmModelList(currentProvider, currentModel);
        updateChatLlmModelList(currentProvider, currentModel);
        
        // Base URL 표시 (커스텀 provider인 경우)
        updateLlmBaseUrlVisibility(currentProvider);
        if (result.llm_base_url) {
            const baseUrlInput = document.getElementById('llmBaseUrlInput');
            if (baseUrlInput) baseUrlInput.value = result.llm_base_url;
        }
        
        // 시스템 프롬프트 로드
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
        
        console.log('LLM 프로바이더 로드 완료:', Object.keys(llmProviders).length + '개');
    } catch (error) {
        console.error('LLM 프로바이더 로드 실패:', error);
    }
}

function updateLlmModelList(providerId, currentModel = '') {
    const modelSelect = document.getElementById('llmModelSelect');
    const customInput = document.getElementById('llmModelCustomInput');
    if (!modelSelect || !llmProviders[providerId]) return;
    
    const provider = llmProviders[providerId];
    modelSelect.innerHTML = '<option value="">기본 모델</option>';
    
    // 프리셋 모델 추가
    provider.models.forEach(model => {
        const opt = document.createElement('option');
        opt.value = model;
        opt.textContent = model;
        modelSelect.appendChild(opt);
    });
    
    // 직접 입력 옵션 추가
    const customOpt = document.createElement('option');
    customOpt.value = '__custom__';
    customOpt.textContent = '✏️ 직접 입력...';
    modelSelect.appendChild(customOpt);
    
    // 현재 모델이 프리셋에 있는지 확인
    const isPresetModel = currentModel === '' || provider.models.includes(currentModel);
    
    if (isPresetModel) {
        modelSelect.value = currentModel;
        if (customInput) customInput.style.display = 'none';
    } else {
        // 프리셋에 없으면 직접 입력 모드
        modelSelect.value = '__custom__';
        if (customInput) {
            customInput.style.display = 'block';
            customInput.value = currentModel;
        }
    }
    
    // 기본 모델 표시
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

// 대화탭 LLM 모델 목록 업데이트
function updateChatLlmModelList(providerId, currentModel = '') {
    const modelSelect = document.getElementById('chatLlmModelSelect');
    if (!modelSelect || !llmProviders[providerId]) return;
    
    const provider = llmProviders[providerId];
    modelSelect.innerHTML = '<option value="">기본</option>';
    
    provider.models.forEach(model => {
        const opt = document.createElement('option');
        opt.value = model;
        // 모델명이 길면 줄임 (대화탭에서는 공간이 좁음)
        opt.textContent = model.length > 20 ? model.substring(0, 18) + '...' : model;
        opt.title = model;  // 전체 이름은 툴팁으로
        if (model === currentModel) opt.selected = true;
        modelSelect.appendChild(opt);
    });
}

// 대화탭에서 LLM 설정 변경 시 자동 저장
async function saveChatLlmSettings() {
    const provider = document.getElementById('chatLlmProviderSelect')?.value;
    const model = document.getElementById('chatLlmModelSelect')?.value;
    
    if (!provider) return;
    
    try {
        await apiCall('/settings', 'POST', {
            llm_provider: provider,
            llm_model: model
        });
        
        // 설정탭 셀렉트도 동기화
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
    if (baseUrlGroup) {
        // 커스텀 provider이거나 로컬 서버인 경우 Base URL 표시
        baseUrlGroup.style.display = 
            (providerId === 'custom' || providerId === 'ollama' || providerId === 'lmstudio') 
            ? 'block' : 'none';
    }
}

async function saveLlmSettings() {
    const provider = document.getElementById('llmProviderSelect').value;
    const apiKey = document.getElementById('llmApiKeyInput').value.trim();
    const baseUrl = document.getElementById('llmBaseUrlInput').value.trim();
    
    // 모델: 직접 입력인 경우 customInput 값 사용
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
        
        // 대화탭 셀렉트도 동기화
        const chatProviderSelect = document.getElementById('chatLlmProviderSelect');
        const chatModelSelect = document.getElementById('chatLlmModelSelect');
        if (chatProviderSelect) chatProviderSelect.value = provider;
        if (chatModelSelect) updateChatLlmModelList(provider, model);
        
        addMessage('system', `✅ LLM 설정 저장됨 (${llmProviders[provider]?.name || provider}${model ? ' / ' + model : ''})`);
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

// 레거시 호환
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

// 시스템 프롬프트 저장
async function saveSystemPrompts() {
    const translatePrompt = document.getElementById('translateSystemPrompt')?.value || '';
    const enhancePrompt = document.getElementById('enhanceSystemPrompt')?.value || '';
    
    try {
        await apiCall('/settings', 'POST', {
            translate_system_prompt: translatePrompt,
            enhance_system_prompt: enhancePrompt
        });
        addMessage('system', '✅ 시스템 프롬프트 저장됨');
    } catch (error) {
        addMessage('system', `❌ 저장 실패: ${error.message}`, 'error');
    }
}

// 번역 시스템 프롬프트 초기화
function resetTranslatePrompt() {
    const translatePromptInput = document.getElementById('translateSystemPrompt');
    if (translatePromptInput && defaultTranslatePrompt) {
        translatePromptInput.value = defaultTranslatePrompt;
        addMessage('system', '✅ 번역 시스템 프롬프트가 기본값으로 초기화되었습니다.');
    }
}

// 향상 시스템 프롬프트 초기화
function resetEnhancePrompt() {
    const enhancePromptInput = document.getElementById('enhanceSystemPrompt');
    if (enhancePromptInput && defaultEnhancePrompt) {
        enhancePromptInput.value = defaultEnhancePrompt;
        addMessage('system', '✅ 향상 시스템 프롬프트가 기본값으로 초기화되었습니다.');
    }
}

// ============= UI 헬퍼 =============
function switchTab(tabId) {
    // 네비게이션 버튼 상태
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    
    // 탭 컨텐츠 표시
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.toggle('active', tab.id === `tab-${tabId}`);
    });
    
    // 탭별 데이터 로드
    if (tabId === 'gallery') loadGallery();
    if (tabId === 'history') loadHistory();
    if (tabId === 'favorites') loadFavorites();
}

function showImageModal(path, metadata) {
    const modal = document.getElementById('imageModal');
    const img = document.getElementById('modalImage');
    const info = document.getElementById('modalInfo');
    
    img.src = path;
    
    if (metadata) {
        let infoText = '';
        if (metadata.prompt) infoText += `프롬프트: ${metadata.prompt}\n`;
        if (metadata.seed) infoText += `시드: ${metadata.seed}\n`;
        if (metadata.width && metadata.height) infoText += `해상도: ${metadata.width}×${metadata.height}\n`;
        if (metadata.steps) infoText += `스텝: ${metadata.steps}\n`;
        info.textContent = infoText;
    } else {
        info.textContent = '';
    }
    
    modal.classList.add('active');
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
    // WebSocket 연결
    connectWebSocket();
    
    // 초기 데이터 로드
    updateModelStatus();
    loadTemplates();
    loadQuantizationOptions();
    loadLlmProviders();
    
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
        // Enter 키로 생성 (번역 후 자동 생성)
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
    
    // CPU 오프로딩 체크박스 동기화
    const chatCpuCheck = document.getElementById('chatCpuOffloadCheck');
    const settingsCpuCheck = document.getElementById('cpuOffloadCheck');
    if (chatCpuCheck && settingsCpuCheck) {
        chatCpuCheck.addEventListener('change', (e) => {
            settingsCpuCheck.checked = e.target.checked;
        });
        settingsCpuCheck.addEventListener('change', (e) => {
            chatCpuCheck.checked = e.target.checked;
        });
    }
    
    // 양자화 선택 드롭다운 동기화
    const chatQuantSelect = document.getElementById('chatQuantizationSelect');
    const settingsQuantSelect = document.getElementById('quantizationSelect');
    if (chatQuantSelect && settingsQuantSelect) {
        chatQuantSelect.addEventListener('change', (e) => {
            settingsQuantSelect.value = e.target.value;
        });
        settingsQuantSelect.addEventListener('change', (e) => {
            chatQuantSelect.value = e.target.value;
        });
    }
    
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
            saveChatLlmSettings();
        });
    }
    
    const chatLlmModelSelect = document.getElementById('chatLlmModelSelect');
    if (chatLlmModelSelect) {
        chatLlmModelSelect.addEventListener('change', () => {
            saveChatLlmSettings();
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
    
    // 레거시 호환
    const btnSaveApiKey = document.getElementById('btnSaveApiKey');
    if (btnSaveApiKey) {
        btnSaveApiKey.addEventListener('click', saveApiKey);
    }
    
    // 갤러리
    document.getElementById('btnRefreshGallery').addEventListener('click', loadGallery);
    
    // 히스토리
    document.getElementById('btnClearHistory').addEventListener('click', clearHistory);
    
    // 즐겨찾기
    document.getElementById('btnSaveFavorite').addEventListener('click', saveFavorite);
    
    // 모달 닫기
    document.getElementById('closeImageModal').addEventListener('click', () => closeModal('imageModal'));
    
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
    
    // 즐겨찾기 저장 버튼 (프롬프트 입력 후 ⭐ 버튼 추가)
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
});
