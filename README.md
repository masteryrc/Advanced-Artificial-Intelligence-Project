# Advanced-Artificial-Intelligence-Project
This project is used to submit the final project for the Advanced Artificial Intelligence course.

database user：
CREATE USER 'bankUser'@'localhost' IDENTIFIED BY 'Bank_User123';
database name：
bank_database

CREATE TABLE pay_table (
    id INT PRIMARY KEY AUTO_INCREMENT,   -- 自增主键
    cardnum VARCHAR(30) NOT NULL,        -- 卡号，字符串类型，不允许为空
    amount DECIMAL(10,2) NOT NULL,       -- 金额，保留两位小数，不允许为空
    date DATE NOT NULL                   -- 日期，仅日期部分，不允许为空
);

main.py--用于主智能体启动。
knowledge_base/search.py--用于知识库向量化。