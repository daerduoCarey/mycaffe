#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/dataset_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;

struct bbox_t {
  float x, y, width, height;
};

DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb", "The backend for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_string(bounding_boxes, "", "Bounding box files");
DEFINE_string(images, "", "Image manifest");
DEFINE_string(image_class_labels, "", "Image class labels");
DEFINE_string(train_test_split, "", "Train/test split");
DEFINE_string(image_directory, "", "Base image directory");
DEFINE_string(train_output, "", "Train output path");
DEFINE_string(test_output, "", "Test output path");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_bounding_boxes.empty());
  CHECK(!FLAGS_images.empty());
  CHECK(!FLAGS_image_class_labels.empty());
  CHECK(!FLAGS_train_test_split.empty());
  CHECK(!FLAGS_image_directory.empty());
  CHECK(!FLAGS_train_output.empty());
  CHECK(!FLAGS_test_output.empty());

  vector<int> keys;

  std::ifstream bbox_file(FLAGS_bounding_boxes.c_str());
  CHECK(bbox_file);
  std::map<int, bbox_t> bboxes;
  int index;
  bbox_t bbox;
  while (bbox_file >> index >> bbox.x >> bbox.y >> bbox.width >> bbox.height) {
    bboxes[index] = bbox;
    keys.push_back(index);
  }
  bbox_file.close();

  std::ifstream image_file(FLAGS_images.c_str());
  CHECK(image_file);
  std::map<int, string> images;
  string filename;
  while (image_file >> index >> filename) {
    images[index] = filename;
  }
  image_file.close();

  std::ifstream image_class_label_file(FLAGS_image_class_labels.c_str());
  CHECK(image_class_label_file);
  std::map<int, int> image_class_labels;
  int label;
  while (image_class_label_file >> index >> label) {
    image_class_labels[index] = label;
  }
  image_class_label_file.close();

  std::ifstream train_test_file(FLAGS_train_test_split.c_str());
  CHECK(train_test_file);
  std::map<int, bool> train_test;
  bool train;
  while (train_test_file >> index >> train) {
    train_test[index] = train;
  }
  train_test_file.close();


  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(keys.begin(), keys.end());
  }
  LOG(INFO) << "A total of " << keys.size() << " images.";

  const std::string& db_backend = FLAGS_backend;

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Open new db
  shared_ptr<Dataset<string, Datum> > train_dataset =
      DatasetFactory<string, Datum>(db_backend);
  shared_ptr<Dataset<string, Datum> > test_dataset =
      DatasetFactory<string, Datum>(db_backend);


  // Open db
  CHECK(train_dataset->open(FLAGS_train_output, Dataset<string, Datum>::New));
  CHECK(test_dataset->open(FLAGS_test_output, Dataset<string, Datum>::New));

  // Storing to db
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  for (int key_id = 0; key_id < keys.size(); ++key_id) {
    int key = keys.at(key_id);
    bool train = train_test.at(key);
    const string& filename = images.at(key);
    int label = image_class_labels.at(key);
    const bbox_t& bbox = bboxes.at(key);

    shared_ptr<Dataset<string, Datum> > dataset = train ? train_dataset :
        test_dataset;

    cv::Mat image = cv::imread(FLAGS_image_directory + "/" + filename);
    int x1 = std::max(0.0, bbox.x - 0.1 * bbox.width);
    int x2 = std::min(image.cols - 1.0, bbox.x + 1.1 * bbox.width);
    int y1 = std::max(0.0, bbox.y - 0.1 * bbox.height);
    int y2 = std::min(image.rows - 1.0, bbox.y + 1.1 * bbox.height);
    image = image.colRange(x1, x2).rowRange(y1, y2);
    cv::resize(image, image, cv::Size(resize_width, resize_height));
    CVMatToDatum(image, &datum);
    datum.set_label(label - 1);

    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", key_id,
        filename.c_str());

    // Put in db
    CHECK(dataset->put(string(key_cstr, length), datum));

    if (++count % 1000 == 0) {
      // Commit txn
      CHECK(dataset->commit());
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    CHECK(train_dataset->commit());
    CHECK(test_dataset->commit());
    LOG(ERROR) << "Processed " << count << " files.";
  }
  train_dataset->close();
  test_dataset->close();
  return 0;
}
