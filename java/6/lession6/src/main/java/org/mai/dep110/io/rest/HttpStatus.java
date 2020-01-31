package org.mai.dep110.io.rest;

public enum HttpStatus {
    NOT_FOUND(404),
    REDIRECT(302),
    SERVER_ERROR(500),
    OK(200)
    ;

    private int code;

    public int getCode() {
        return code;
    }

    HttpStatus(int code) {
        this.code = code;
    }

}
