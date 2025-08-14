let floatingWindow = null;

chrome.runtime.onMessage.addListener((message) => {
    switch (message.action) {
        case 'start':
            if (!floatingWindow) {
                // 打開懸浮視窗
                chrome.windows.create({
                    url: chrome.runtime.getURL('floating.html'),
                    type: 'popup',
                    width: 300,
                    height: 400
                }, (window) => {
                    floatingWindow = window.id;
                });
            }
            break;

        case 'stop':
            if (floatingWindow) {
                // 關閉懸浮視窗
                chrome.windows.remove(floatingWindow);
                floatingWindow = null;
            }
            break;

        case 'updatePoseData':
            if (floatingWindow) {
                chrome.tabs.query({ windowId: floatingWindow }, (tabs) => {
                    if (tabs.length > 0) {
                        chrome.tabs.sendMessage(tabs[0].id, { action: 'updatePoseData', data: message.data });
                    }
                });
            }
            break;
    }
});
