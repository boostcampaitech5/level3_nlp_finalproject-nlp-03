def log():
    import logging

    # 1 logger instance
    logger = logging.getLogger(__name__)

    # 2 formatter
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 3 handler
    streamHandler = logging.StreamHandler() # 콘솔 출력용
    fileHandler = logging.FileHandler("server.log") # 파일 기록용

    # 4 Logger instance에 formatter 설정
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # 5 Logger Instance에 handler추가
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    # 6 log level
    logger.setLevel(level=logging.DEBUG)

    return logger

# 7 선 호출 - 그 이후에 logger.debug로 기록 가능
if __name__=="__main__":
    logger=log()
    logger.debug("hello")