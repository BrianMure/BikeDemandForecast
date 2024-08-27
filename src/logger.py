#any execution that happens we should be able to log all that information, the execution so we can be able to track any error that may occur
import logging
import os
from datetime import datetime

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", log_file)
os.makedirs(logs_path, exist_ok = True)  # Create the logs directory if it doesn't exist

log_file_path = os.path.join(logs_path, log_file)

log_format = " [%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

#configure the root logger
logging.basicConfig(
        filename = log_file_path,
        format=log_format,
        level = logging.INFO,
)

