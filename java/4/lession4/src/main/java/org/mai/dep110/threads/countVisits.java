package org.mai.dep110.threads;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


public class countVisits {

    public static void main(String[] args) throws InterruptedException {
        Site c = new Site();
        c.setPages();

        ExecutorService service = null;
        try {
            service = Executors.newFixedThreadPool(50);
                    for (int j = 0; j < 10; j++) {
                        for (int i = 0; i < 50; i++) {
                            for (Page key : c.getPages()) {
                            service.submit(key::visit);
                    }
                }
            }

        } finally {
            if (null != service) service.shutdown();
        }

        synchronized (c) {
            c.notifyAll();
        }

        while (!service.awaitTermination(24L, TimeUnit.HOURS)) {
            System.out.println("Not yet. Still waiting for termination");
        }

        c.printHit();
    }

}

class Site {
    private List<Page> pages = new ArrayList<>();

    public void setPages() {
        pages.add(new Page("a"));
        pages.add(new Page("b"));
        pages.add(new Page("c"));
        pages.add(new Page("d"));
        pages.add(new Page("e"));
    }

    public void printHit() {
        for (Page entry : pages) {
            System.out.println(entry.toString());
        }
    }

    public List<Page> getPages() { return pages; }

}

class Page {

    private int cnt;
    private String page_name;

    public Page(String page_name) {
        this.cnt = 0;
        this.page_name = page_name;
    }

    ;

    public int getCount() {
        return cnt;
    }


    public String getName() {
        return page_name;
    }


    public synchronized void visit() {
//        try {
//            wait(1000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        cnt = cnt + 1;
        System.out.println("Page " + page_name + " count " + cnt +  " by " + Thread.currentThread().getName());
//        notifyAll();
    }

    @Override
    public String toString() {
        return "Page name " + getName() + " page count " + getCount();
    }
}




//old version
//pckage org.mai.dep110.threads;
//
//        import java.util.HashMap;
//        import java.util.Map;
//        import java.util.concurrent.ExecutorService;
//        import java.util.concurrent.Executors;
//        import java.util.concurrent.TimeUnit;
//
//
//public class countVisits {
//
//    public static void main(String[] args) throws InterruptedException {
//        Counter c = new Counter();
//        c.setHit();
//
//        ExecutorService service = null;
//        try {
//            service = Executors.newFixedThreadPool(50);
//
//            for (String key : c.getHit().keySet()) {
//                for (int j = 0; j < 10; j++) {
//                    for (int i = 0; i < 50; i++) {
//                        service.submit(() -> c.synchronized_incriment_with_wait(key));
//                    }
//                }
//            }
//
//        } finally {
//            if (null != service) service.shutdown();
//        }
//
//        synchronized (c) {
//            c.notifyAll();
//        }
//
//        while (!service.awaitTermination(24L, TimeUnit.HOURS)) {
//            System.out.println("Not yet. Still waiting for termination");
//        }
//
//        c.printHit();
//    }
//
//}
//
//class Counter {
//    private Map<String, Integer> hits = new HashMap<String, Integer>();
//
//    public void setHit() {
//        hits.put("a", 0);
//        hits.put("b", 0);
//        hits.put("c", 0);
//        hits.put("d", 0);
//        hits.put("e", 0);
//    }
//
//    public void printHit() {
//        for (Map.Entry<String, Integer> entry : hits.entrySet()) {
//            System.out.println(entry.getKey() + " :: " + entry.getValue());
//        }
//    }
//
//    public Map<String, Integer> getHit() { return hits; }
//
//    public void incriment(String key) { hits.put(key, hits.get(key) + 1); }
//
//    public synchronized void synchronized_incriment(String key) { hits.put(key, hits.get(key) + 1); }
//
//    //когда делаем синк на метод, он сделает синк на весь объект, то есть по факту получится однопоточное приложение, так как каждый
//    //вызов синк одним потоком будет блочить весь коунтер и они будут ждать. по факту получится однопоточное приложение
//    public synchronized void synchronized_incriment_with_wait(String key) {
//        try {
//            wait(1000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//        hits.put(key, hits.get(key) + 1);
//        System.out.println(key + " = " + hits.get(key) + " by " +Thread.currentThread().getName());
//        notifyAll();
//    }
//
//}
