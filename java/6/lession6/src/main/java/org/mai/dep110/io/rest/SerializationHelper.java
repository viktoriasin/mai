package org.mai.dep110.io.rest;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.log4j.Logger;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class SerializationHelper<T extends Serializable> {

    Class<T> serialazationType;

    public SerializationHelper(Class<T> serialazationType) {
        this.serialazationType = serialazationType;
    }

    private Logger log = Logger.getLogger(getClass());

    ObjectMapper mapper = new ObjectMapper();


    /*
      Необходимо десереализовать объект из файла по указанному пути
     */
    public T loadFromFile(String path) {
        T objRestored = null;
        try (ObjectInputStream objectInputStream = new ObjectInputStream(
                new GZIPInputStream(
                        new FileInputStream(path))))  {
            objRestored = (T) objectInputStream.readObject();
        } catch (ClassNotFoundException e){
        e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return objRestored;
    }

    /*
      Необходимо сохранить сереализованный объект в файл по указанному пути
     */
    public boolean saveToFile(String path, T toSave) {
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(
                new GZIPOutputStream(
                        new FileOutputStream(path)))) {
            objectOutputStream.writeObject(toSave);
        } catch(IOException e){
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public String convertToJsonString(T toConvert) {
        try {
            String json = mapper.writeValueAsString(toConvert);
            return json;
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }

        return null;
    }

    public void writeJsonToStream(OutputStream output, T toWrite) {
        try {
            mapper.writeValue(output, toWrite);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public T parseJson(String json) {
        try {
            return mapper.readValue(json, serialazationType);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
