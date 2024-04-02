const mysql = require('mysql2')

const pool = mysql.createPool({
    // mysql authentication info 
    host: "192.168.1.53",
    user: "mysql",
    port: 3306,
    password: "1234",
    database: "testdb"
})

const promisePool = pool.promise()

module.exports = promisePool;
