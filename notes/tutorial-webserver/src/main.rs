// 面白そうなチュートリアルがあったのでやってみる
// see: https://doc.rust-lang.org/book/ch20-01-single-threaded.html
// test command: nc 127.0.0.1 8137

use std::{
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
};

use tutorial_webserver::ThreadPool;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:8137").unwrap();
    let pool = ThreadPool::new(4);

    for stream in listener.incoming().take(2) {
        let stream = stream.unwrap();

        // handle_connection(stream);
        // thread::spawn(|| handle_connection(stream));
        pool.execute(|| handle_connection(stream));
    }

    println!("Shutting down.");
}

fn handle_connection(mut stream: TcpStream) {
    let buf_reader = BufReader::new(&mut stream);
    let http_request: Vec<_> = buf_reader
        .lines()
        .map(|result| result.unwrap())
        .take_while(|line| !line.is_empty())
        .collect();

    let response = "HTTP/1.1 200 OK\r\n\r\n";

    stream.write_all(response.as_bytes()).unwrap();

    println!("Request: {:#?}", http_request);
}
