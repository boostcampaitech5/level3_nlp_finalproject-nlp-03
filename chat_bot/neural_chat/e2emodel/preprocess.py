def format_scenario(input: dict, sep_token: str) -> str:
    """
    모델이 학습할 수 있는 형태로 시나리오를 변환합니다.
    """
    title = f"제목: {input['title']}"
    desc = f"상품 설명: {input['description']}"
    price = f"가격: {input['price']}"

    return sep_token.join([title, desc, price]) + sep_token


def format_chat(input: dict, sep_token: str) -> str:
    """
    모델이 학습할 수 있는 형태로 채팅을 변환합니다.
    """

    return (
        sep_token.join(
            [f"{item['role']}: {item['message']}" for item in input["events"]]
        )
        + sep_token
    )
