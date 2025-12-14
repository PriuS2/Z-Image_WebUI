/**
 * Z-Image WebUI - 인증 페이지 JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initForms();
    initPasswordToggle();
    initGuestLogin();
});

/**
 * 탭 초기화
 */
function initTabs() {
    const tabs = document.querySelectorAll('.auth-tab');
    const forms = document.querySelectorAll('.auth-form');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;
            
            // 탭 활성화
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // 폼 활성화
            forms.forEach(form => {
                form.classList.remove('active');
                if (form.id === `${targetTab}Form`) {
                    form.classList.add('active');
                }
            });
            
            // 에러 메시지 초기화
            clearErrors();
        });
    });
}

/**
 * 폼 초기화
 */
function initForms() {
    // 로그인 폼
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
    
    // 회원가입 폼
    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        registerForm.addEventListener('submit', handleRegister);
    }
}

/**
 * 비밀번호 표시/숨기기 토글
 */
function initPasswordToggle() {
    const toggleButtons = document.querySelectorAll('.toggle-password');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const input = button.parentElement.querySelector('input');
            const icon = button.querySelector('i');
            
            if (input.type === 'password') {
                input.type = 'text';
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
            } else {
                input.type = 'password';
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
            }
        });
    });
}

/**
 * 로그인 처리
 */
async function handleLogin(e) {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = form.querySelector('.auth-btn');
    const errorDiv = document.getElementById('loginError');
    
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    
    // 아이디와 비밀번호가 모두 비어있으면 게스트 로그인
    if (!username && !password) {
        handleGuestLogin();
        return;
    }
    
    // 둘 중 하나만 비어있으면 에러
    if (!username || !password) {
        showError(errorDiv, '아이디와 비밀번호를 입력해주세요.');
        return;
    }
    
    // 버튼 비활성화
    setButtonLoading(submitBtn, true);
    clearErrors();
    
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // 로그인 성공 - 메인 페이지로 이동
            window.location.href = '/';
        } else {
            showError(errorDiv, data.detail || data.message || '로그인에 실패했습니다.');
        }
    } catch (error) {
        console.error('Login error:', error);
        showError(errorDiv, '서버와 통신 중 오류가 발생했습니다.');
    } finally {
        setButtonLoading(submitBtn, false);
    }
}

/**
 * 회원가입 처리
 */
async function handleRegister(e) {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = form.querySelector('.auth-btn');
    const errorDiv = document.getElementById('registerError');
    
    const username = document.getElementById('registerUsername').value.trim();
    const password = document.getElementById('registerPassword').value;
    const passwordConfirm = document.getElementById('registerPasswordConfirm').value;
    
    // 유효성 검사
    if (!username || !password || !passwordConfirm) {
        showError(errorDiv, '모든 항목을 입력해주세요.');
        return;
    }
    
    if (password !== passwordConfirm) {
        showError(errorDiv, '비밀번호가 일치하지 않습니다.');
        return;
    }
    
    if (username.length < 3) {
        showError(errorDiv, '아이디는 3자 이상이어야 합니다.');
        return;
    }
    
    if (password.length < 4) {
        showError(errorDiv, '비밀번호는 4자 이상이어야 합니다.');
        return;
    }
    
    // 버튼 비활성화
    setButtonLoading(submitBtn, true);
    clearErrors();
    
    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                password,
                password_confirm: passwordConfirm
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // 회원가입 성공 - 로그인 탭으로 전환 및 메시지 표시
            alert('회원가입이 완료되었습니다. 로그인해주세요.');
            
            // 로그인 탭으로 전환
            document.querySelector('.auth-tab[data-tab="login"]').click();
            
            // 아이디 자동 입력
            document.getElementById('loginUsername').value = username;
            document.getElementById('loginPassword').focus();
            
            // 회원가입 폼 초기화
            form.reset();
        } else {
            showError(errorDiv, data.detail || data.message || '회원가입에 실패했습니다.');
        }
    } catch (error) {
        console.error('Register error:', error);
        showError(errorDiv, '서버와 통신 중 오류가 발생했습니다.');
    } finally {
        setButtonLoading(submitBtn, false);
    }
}

/**
 * 에러 메시지 표시
 */
function showError(element, message) {
    if (element) {
        element.textContent = message;
        element.classList.add('show');
    }
}

/**
 * 에러 메시지 초기화
 */
function clearErrors() {
    const errors = document.querySelectorAll('.form-error');
    errors.forEach(error => {
        error.textContent = '';
        error.classList.remove('show');
    });
}

/**
 * 버튼 로딩 상태 설정
 */
function setButtonLoading(button, loading) {
    if (!button) return;
    
    if (loading) {
        button.classList.add('loading');
        button.disabled = true;
        const icon = button.querySelector('i');
        if (icon) {
            icon.dataset.originalClass = icon.className;
            icon.className = 'fas fa-spinner';
        }
    } else {
        button.classList.remove('loading');
        button.disabled = false;
        const icon = button.querySelector('i');
        if (icon && icon.dataset.originalClass) {
            icon.className = icon.dataset.originalClass;
        }
    }
}

/**
 * 게스트 로그인 초기화
 */
function initGuestLogin() {
    const guestBtn = document.getElementById('guestLoginBtn');
    console.log('initGuestLogin called, button:', guestBtn);
    if (guestBtn) {
        guestBtn.addEventListener('click', (e) => {
            console.log('Guest button clicked');
            e.preventDefault();
            e.stopPropagation();
            handleGuestLogin();
        });
    }
}

/**
 * 게스트 로그인 처리
 */
async function handleGuestLogin() {
    const guestBtn = document.getElementById('guestLoginBtn');
    const errorDiv = document.getElementById('loginError');
    
    // 버튼 비활성화
    setButtonLoading(guestBtn, true);
    clearErrors();
    
    try {
        const response = await fetch('/api/auth/guest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // 게스트 로그인 성공 - 메인 페이지로 이동
            window.location.href = '/';
        } else {
            showError(errorDiv, data.detail || data.message || '게스트 로그인에 실패했습니다.');
        }
    } catch (error) {
        console.error('Guest login error:', error);
        showError(errorDiv, '서버와 통신 중 오류가 발생했습니다.');
    } finally {
        setButtonLoading(guestBtn, false);
    }
}

