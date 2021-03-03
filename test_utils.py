import sys
sys.path.append("data_augmentation/src")
import utils

def test_sample_compositions_by_species():
    path = "../TAKAO_BIRD_WAV_feb20_augmented_samples-0.33n-0.33ts-0.33w-exc/spectrograms_augmented/"
    df = utils.sample_compositions_by_species(path, True)
    print(df)

if __name__ == '__main__':
    test_sample_compositions_by_species()
