var mysql = require('sync-mysql');
const env = require('dotenv').config({ path: "../../.env" });

var connection = new mysql({
    host: process.env.host,
    user: process.env.user,
    port: process.env.port,
    password: process.env.password,
    database: process.env.database
});

// select from st_info table
let result = connection.query('select * from st_info');
console.log(result);

// make insert data
let data = {
    st_id: "202499",
    name: "Moon",
    dept: "Computer"
}

// inserted data's id
let insertId = data.st_id;

//insert query
result = connection.query("insert into st_info values (?, ?, ?)",
    [insertId, data.name, data.dept]);
console.log("\nData is inserted~!!");

// select * query for inserted data
result = connection.query("select * from st_info where st_id=?", [insertId]);
console.log(result);

// update query
result = connection.query("update st_info set dept=? where st_id=?", ["Game", insertId]);
console.log("\nData is Updated~!!");

// select * query for updated data
result = connection.query("select * from st_info where st_id=?", [insertId]);
console.log(result);

// delete query
result = connection.query("delete from st_info where st_id=?", [insertId]);
console.log('\nData is Deleted~!!');

// select * query for updated data
result = connection.query("select * from st_info");
console.log(result);
