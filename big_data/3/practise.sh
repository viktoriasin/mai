#скопировать все файлы в hdfs из папки кроме одного

ls /home/my_directory | grep -v 'doNotCopy.txt' | while read -r fileName ; do 
    eval "hdfs dfs -copyFromLocal -f $fileName /path/to/HDFS/" 
done

#... кроме нескольких 
ls /home/my_directory | grep -v 'doNotCopy.txt\|dirDoNotCopy\|anotherTextFile.txt' | while read -r fileName ; do 
    eval "hdfs dfs -copyFromLocal -f $fileName /path/to/HDFS/" 
done

#посмотреть файлы в hdfs по пути

hdfs dfs -ls -C /path/to/HDFS/

 #скопировать несколько файлов одну и туже директорию 
 hadoop fs -cp /path1/file1 /path2/file2 path3/file3 /pathx/target



hadoop fs -mkdir <paths>

#Copy a file from/To Local file system to HDFS
hadoop fs -copyFromLocal <localsrc> URI

hadoop fs -copyFromLocal /home/saurzcode/abc.txt  /user/saurzcode/abc.txt


#Extract zip file into a new or existing directory dir1:

7z x file.zip -o./dir1


#Compress directory dir1 to a new zip file:

7z a newfile.zip ./dir1

tar -cvf lab7.tar lab7

zip lab7.zip lab7/*

rar a lab7.rar lab7/*