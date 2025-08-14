let floatingWindow = null;

chrome.runtime.onMessage.addListener((message) => {
    switch (message.action) {
        case 'start':
            if (!floatingWindow) {
                chrome.windows.create({
                    url: chrome.runtime.getURL('floating.html'),
                    type: 'popup',
                    width: 300,
                    height: 400
                }, (window) => {
                    if (window) {
                        floatingWindow = window.id;
                        chrome.storage.local.set({ running: true });
                    }
                });
            }
            break;

        case 'stop':
            if (floatingWindow !== null) {
                // 檢查視窗是否還存在
                chrome.windows.get(floatingWindow, {}, (win) => {
                    if (chrome.runtime.lastError || !win) {
                        // 視窗不存在，直接清理狀態
                        floatingWindow = null;
                        chrome.storage.local.set({ running: false });
                        chrome.runtime.sendMessage({ action: 'floatingClosed' }).catch(() => {});
                    } else {
                        // 視窗還存在，可以安全移除
                        chrome.windows.remove(floatingWindow, () => {
                            floatingWindow = null;
                            chrome.storage.local.set({ running: false });
                            chrome.runtime.sendMessage({ action: 'floatingClosed' }).catch(() => {});
                        });
                    }
                });
            } else {
                // 沒有視窗，但仍需確保狀態正確
                chrome.storage.local.set({ running: false });
                chrome.runtime.sendMessage({ action: 'floatingClosed' }).catch(() => {});
            }
            break;

        case 'updatePoseData':
            if (floatingWindow !== null) {
                chrome.tabs.query({ windowId: floatingWindow }, (tabs) => {
                    if (tabs.length > 0) {
                        chrome.tabs.sendMessage(tabs[0].id, {
                            action: 'updatePoseData',
                            data: message.data
                        });
                    }
                });
            }
            break;
    }
});

// 懸浮視窗被手動關閉時的處理
chrome.windows.onRemoved.addListener((closedWindowId) => {
    if (closedWindowId === floatingWindow) {
        floatingWindow = null;
        chrome.storage.local.set({ running: false });

        chrome.runtime.sendMessage({ action: 'floatingClosed' }).catch(() => {
            console.warn('Popup not open, skipping message');
        });
    }
});
