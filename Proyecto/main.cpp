#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

#include <stdio.h>

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>


using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

using namespace std;


static void construct_net(network<sequential>& nn) {


    // by default  backend_t::tiny_dnn
    core::backend_t backend_type = core::default_engine();



    nn << convolutional_layer(48, 48, 5, 1,
                                      16,  // C1, 1@32x32-in, 6@28x28-out
                                      padding::valid, true, 1, 1, backend_type)
               << relu_layer(44, 44, 16)
               << max_pooling_layer(44, 44, 16,
                                        2)  // S2, 6@28x28-in, 6@14x14-out
               << relu_layer(22, 22, 16)

               << fully_connected_layer(7744, 3, true,  // F6, 120-in, 10-out
                                        backend_type);

}

static void train_lenet(std::vector<label_t> & train_labels,std::vector<label_t>& test_labels,std::vector<vec_t>& train_images,std::vector<vec_t>& test_images) {
    // loss-function y learning strategy
    network<sequential> nn;
    adagrad optimizer;

    construct_net(nn);
    std::cout << "load models..." << std::endl;


    std::cout << "start training" << std::endl;

    progress_display disp(static_cast<unsigned long>(train_images.size()));
    timer t;
    int minibatch_size = 10;
    int num_epochs = 100;

    optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));


    // crear callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;
        disp.restart(static_cast<unsigned long>(train_images.size()));
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test y results"data"
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save network model & trained weights
    nn.save("netMIAS");
}


// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
    Activation a;
    return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string& imagefilename,
    double minv,
    double maxv,
    int w,
    int h,
    vec_t& data) {

    image<> img(imagefilename, image_type::grayscale);
    image<> resized = resize_image(img, w, h);
    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
        [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void recognize(const std::string& dictionary, const std::string& filename) {
    network<sequential> nn;

    nn.load(dictionary);

    // convert imagefile to vec_t
    vec_t data;
    convert_image(filename, -1.0, 1.0, 48, 48, data);

    // recognize
    auto res = nn.predict(data);
    vector<pair<double, int> > scores;

    // sort & print top-3
    for (int i = 0; i < 3; i++)
        scores.emplace_back(rescale<activation::tanh>(res[i]), i);

    sort(scores.begin(), scores.end(), greater<pair<double, int>>());

    for (int i = 0; i < 3; i++)
        cout << scores[i].second << "," << scores[i].first << endl;

    // save outputs of each layer
    for (size_t i = 0; i < nn.depth(); i++) {
        auto out_img = nn[i]->output_to_image();
        auto filename = "layer_" + std::to_string(i) + ".png";
        out_img.save(filename);
    }
    // save filter shape of first convolutional layer
    {
        auto weight = nn.at<convolutional_layer>(0).weight_to_image();
        auto filename = "weights.png";
        weight.save(filename);
    }
}


void load_database(vector<vec_t>& train, vector<label_t>& train_label)
{
    //string dir="data/Head1/";
    vector<string> dir ={"all-mias"};


    string line;
    ifstream fileIn("all-mias/info");
    if (fileIn.is_open())
    {
        while ( getline (fileIn,line) )
        {
            std::stringstream pixels(line);
            int index = 0;

            for (std::string dato; std::getline(pixels,dato, ' '); )
            {
                if (index==0){

                    vec_t img;
                    convert_image("all-mias/"+dato+".pgm", -1.0, 1.0, 48, 48, img);
                    train.push_back(img);

                    CImg<unsigned char> img(("all-mias/"+dato+".pgm").c_str());
                    img.display();
                }
                else if (index==3){
                    // 0 Normal
                    // 1 Benigno
                    // 2 Maligno
                    if (dato == "B")
                        train_label.push_back(1);
                    else if (dato == "M")
                        train_label.push_back(2);

                    //cout<<"label : "<<dato<<endl;
                }

                ++index;
            }
            if (index==3){
                train_label.push_back(0);
            }

        }
        fileIn.close();
    }

    else cout << "No se encuentra el archivo..."<<endl;


    cout<<train.size()<<endl;
    cout<<train_label.size()<<endl;
}

int main(int argc, char *argv[])
{



    vector<vec_t> train;
    vector<label_t> train_label;


   load_database(train,train_label);


    train_lenet(train_label,train_label, train,train);

    ///imagenes de prueba
    string list_ima[]={"mdb001.pgm","mdb002.pgm","mdb003.pgm","mdb004.pgm","mdb023.pgm"};
    recognize("netMIAS", (list_ima[i]).c_str());


    return 0;
}
