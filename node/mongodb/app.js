const express = require('express')
const morgan = require('morgan')
const fs = require('fs')
const path = require('path')
const mongoClient = require('mongodb').MongoClient

const app = express()
app.set('port', process.env.PORT || 8000)
app.use(morgan('dev'))

var db;
var databaseUrl = 'mongodb://192.168.1.53:27017'

app.get('/', (req, res) => {
    res.send('Web Server Started~!!')
})

app.get('/things', (req, res) => {
    mongoClient.connect(databaseUrl, function (err, database) {
        if (err != null) {
            res.json({ 'count': 0 })
        } else {
            db = database.db('test')
            db.collection('things').find({}).toArray(function (err, result) {
                if (err) throw err
                console.log('result : ')
                console.log(result)
                res.json(JSON.stringify(result))
            })
        }
    })
})

app.get('/emp', (req, res) => {
    mongoClient.connect(databaseUrl, function (err, database) {
        if (err != null) {
            res.json({ 'count': 0 })
        } else {
            db = database.db('test')
            db.collection('emp').find({}).toArray(function (err, result) {
                if (err) throw err
                console.log('result : ')
                console.log(result)
                res.json(JSON.stringify(result))
            })
        }
    })
})

app.get('/primer', (req, res) => {
    mongoClient.connect(databaseUrl, function (err, database) {
        if (err != null) {
            res.json({ 'count': 0 })
        } else {
            db = database.db('test')
            db.collection('primer').find({}).toArray(function (err, result) {
                if (err) throw err
                console.log('result : ')
                console.log(result)
                res.json(JSON.stringify(result))
            })
        }
    })
})

app.listen(app.get('port'), () => {
    console.log('8000 Port : Server Started~!!')
})
