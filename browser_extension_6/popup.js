document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');

    function updateStatus(isRunning, updateStorage = false) {
        if (isRunning) {
            statusIndicator.classList.add('active');
            statusText.textContent = 'Running';
        } else {
            statusIndicator.classList.remove('active');
            statusText.textContent = 'Not Running';
        }

        if (updateStorage) {
            chrome.storage.local.set({ running: isRunning });
        }
    }

    startBtn.addEventListener('click', () => {
        chrome.runtime.sendMessage({ action: 'start' });
        updateStatus(true, true); // 更新狀態 + 寫入儲存
    });

    stopBtn.addEventListener('click', () => {
        chrome.runtime.sendMessage({ action: 'stop' });
        updateStatus(false, true); // 更新狀態 + 寫入儲存
    });

    chrome.storage.local.get('running', (data) => {
        updateStatus(data.running === true);
    });

    chrome.runtime.onMessage.addListener((message) => {
        if (message.action === 'floatingClosed') {
            updateStatus(false, false); // 只更新顯示，不寫入 storage（background 已處理）
        }

        if (message.action === 'floatingOpened') {
            updateStatus(true, false); // 可選，若 background 有傳此訊息
        }
    });
});
