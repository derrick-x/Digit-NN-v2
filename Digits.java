import java.awt.Canvas;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

public class Digits extends Canvas {
    static double[][] drawing = new double[28][28];
    static final int DIGITS = 10;
    static final int TRAIN_SIZE = 10000;
    static final int BACKPROP_COUNT = 200;
    static final int TEST_SIZE = 100;
    static final double AUGMENT_FACTOR = 0.4;

    static boolean mouseDown = false;
    static double mouseX;
    static double mouseY;

    public static void main(String[] args) throws IOException {
        double[][] target = new double[107751][DIGITS];
        double[][] training = new double[107751][784];
        int index = 0;
        System.out.println("Reading images...");
        for (int i = 0; i < DIGITS; i++) {
            File[] trainFiles = new File("dataset/" + i + "/" + i).listFiles();
            for (int j = 0; j < TRAIN_SIZE / DIGITS; j++) {
                for (int k = 0; k < DIGITS; k++) {
                    target[index][k] = 0;
                }
                target[index][i] = 1;
                BufferedImage image = ImageIO.read(trainFiles[i]);
                while (image == null) {
    
                }
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int rgb = image.getRGB(x, y);
                        int alpha = (rgb >> 24) & 0xFF;
                        int red =   (rgb >> 16) & 0xFF;
                        int green = (rgb >>  8) & 0xFF;
                        int blue =  (rgb      ) & 0xFF;
                        training[index][y * 28 + x] = alpha / 256.0;
                    }
                }
                double[][] matrix = {{Math.random() * AUGMENT_FACTOR + 1 - AUGMENT_FACTOR / 2, Math.random() * AUGMENT_FACTOR - AUGMENT_FACTOR / 2}, {Math.random() * AUGMENT_FACTOR - AUGMENT_FACTOR / 2, Math.random() * AUGMENT_FACTOR + 1 - AUGMENT_FACTOR / 2}};
                training[index] = augment(training[index], 28, matrix, (int) (Math.random() * 14 * AUGMENT_FACTOR - 7 * AUGMENT_FACTOR), (int) (Math.random() * 14 * AUGMENT_FACTOR - 7 * AUGMENT_FACTOR), 0.1);
                index++;
            }
        }
        System.out.println("Training on " + TRAIN_SIZE + " images...");
        NeuralNetwork nn = new NeuralNetwork(2, 784, DIGITS, 16, true);
        nn.stochastic = 10;
        nn.dropoutFactor = 0.3;
        nn.l2Factor = 0.01;
        System.out.print(".");
        for (int i = 0; i < BACKPROP_COUNT; i++) {
            System.out.print(" ");
        }
        System.out.print(".\n ");
        for (int i = 0; i < BACKPROP_COUNT; i++) {
            nn.learn(index, training, target);
            System.out.print("-");
        }
        System.out.println();
        System.out.println("Testing on " + TEST_SIZE + " images...");
        int correct = 0;
        BufferedImage[] demo = new BufferedImage[10];
        double[][] explanations = new double[10][784];
        for (int i = 0; i < DIGITS; i++) {
            File[] testFiles = new File("dataset/" + i + "/" + i).listFiles();
            demo[i] = ImageIO.read(testFiles[0]);
            for (int j = TRAIN_SIZE / DIGITS; j < (TRAIN_SIZE + TEST_SIZE) / DIGITS; j++) {
                BufferedImage image = ImageIO.read(testFiles[j]);
                double[] input = new double[784];
                while (image == null) {
    
                }
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int rgb = image.getRGB(x, y);
                        int alpha = (rgb >> 24) & 0xFF;
                        int red =   (rgb >> 16) & 0xFF;
                        int green = (rgb >>  8) & 0xFF;
                        int blue =  (rgb      ) & 0xFF;
                        input[y * 28 + x] = alpha / 256.0;
                    }
                }

                double[] output = NeuralNetwork.softmax(nn.think(input));
                if (j == TRAIN_SIZE / DIGITS) {
                    explanations[i] = nn.explain(input)[i];
                }
                int prediction = 0;
                for (int k = 0; k < DIGITS; k++) {
                    if (output[k] > output[prediction]) {
                        prediction = k;
                    }
                }
                for (int k = 0; k < DIGITS; k++) {
                    System.out.printf(k + ": %.3f ", output[k]);
                }
                System.out.println();
                System.out.println("Expected: " + i + ", predicted: " + prediction);
                if (prediction == i) {
                    correct++;
                }
            }
        }
        double[] base = NeuralNetwork.softmax(nn.think(new double[784]));
        for (int i = 0; i < DIGITS; i++) {
            System.out.printf(i + ": %.3f ", base[i]);
        }
        System.out.println();
        System.out.println(correct * 100.0 / TEST_SIZE + "% accuracy");
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(700, 700);
        Canvas canvas = new Digits();
        canvas.addKeyListener(new KeyListener() {
            @Override
            public void keyTyped(KeyEvent e) {
                
            }

            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyChar() == ' ') {
                    double[] input = new double[784];
                    for (int y = 0; y < 28; y++) {
                        for (int x = 0; x < 28; x++) {
                            input[y * 28 + x] = drawing[y][x];
                        }
                    }
                    double[] output = NeuralNetwork.softmax(nn.think(input));
                    for (int i = 0; i < 10; i++) {
                        System.out.printf("%d: %.5f\n", i, output[i]);
                    }
                    System.out.println();
                    drawing = new double[28][28];
                }
            }

            @Override
            public void keyReleased(KeyEvent e) {
                
            }

        });
        canvas.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                
            }

            @Override
            public void mousePressed(MouseEvent e) {
                mouseDown = true;
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                mouseDown = false;
            }

            @Override
            public void mouseEntered(MouseEvent e) {
                
            }

            @Override
            public void mouseExited(MouseEvent e) {
                
            }

        });
        frame.add(canvas);
        canvas.repaint();
        frame.setVisible(true);
        while (true) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
            }
            if (mouseDown) {
                mouseX = canvas.getMousePosition().getX();
                mouseY = canvas.getMousePosition().getY();
                canvas.repaint();
            }
        }
    }

    @Override
    public void paint(Graphics g) {
        g.setColor(Color.white);
        g.fillRect(0, 0, 700, 700);
        if (mouseDown) {
            try {
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        drawing[y][x] = Math.max(drawing[y][x], Math.min(1, 1 - Math.hypot(y - (mouseY / 25), x - (mouseX / 25)) / 2));
                    }
                }
            } catch (Exception e) {
            }
        }
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                g.setColor(new Color(255 - (int) (drawing[y][x] * 255), 255 - (int) (drawing[y][x] * 255), 255 - (int) (drawing[y][x] * 255)));
                g.fillRect(x * 25, y * 25, 25, 25);
            }
        }
    }
    public static double[] augment(double[] input, int width, double[][] matrix, int xShift, int yShift, double noise) {
        int height = input.length / width;
        double[][] image = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image[y][x] = input[y * width + x];
            }
        }
        double[][] augmented = new double[image.length][image[0].length];
        for (int y = 0; y < image.length; y++) {
            for (int x = 0; x < image[y].length; x++) {
                try {
                    augmented[(int) (y * matrix[0][0] + x * matrix[0][1]) + yShift][(int) (y * matrix[1][0] + x * matrix[1][1]) + xShift] = image[y][x];
                } catch (ArrayIndexOutOfBoundsException e) {
                    Math.abs(0);
                }
                augmented[y][x] += Math.random() * noise - noise / 2;
                augmented[y][x] = Math.max(0, Math.min(augmented[y][x], 1));
            }
        }
        double[] ret = new double[image.length * width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                ret[y * width + x] = augmented[y][x];
            }
        }
        return ret;
}
    static class Weight implements Comparable<Weight> {
        int index;
        double value;
        public Weight(int i, double v) {
            index = i; value = v;
        }

        @Override
        public int compareTo(Weight o) {
            return value < o.value ? 1 : -1;
        }
    }
}
