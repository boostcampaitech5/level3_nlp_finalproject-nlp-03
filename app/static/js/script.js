function openPopup(productId) {
    const inputText = prompt("Enter Your Nickname:");
    if (inputText !== null) {
        // 부모 창으로 데이터 전달
        window.parent.postMessage({ id: productId, text: inputText }, window.location.origin);
    }
}

function receiveMessage(event) {
    const receivedData = event.data;
    if (receivedData !== null) {
        const { id, text } = receivedData;

        // Check if id is a non-negative integer
        const isIdValid = Number.isInteger(Number(id));

        let url;
        if (isIdValid) {
            url = `/chatting/${id}?name=${encodeURIComponent(text)}`;
        } else {
            url = '/';
        }

        window.location.href = url;
    }
}

// 각 제품별 버튼에 이벤트 리스너 추가
const productButtons = document.querySelectorAll(".product-button");
productButtons.forEach(button => {
    button.addEventListener("click", () => {
        const productId = button.dataset.productId;
        openPopup(productId);
    });
});

// 부모 창으로부터 데이터를 수신하고 리디렉션 수행
window.addEventListener("message", receiveMessage, false);
