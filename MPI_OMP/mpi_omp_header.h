#define INPUT_FILE_NAME "input.txt"
#define OUTPUT_FILE_NAME "output.txt"
#define MAX_SEQ_1 3000
#define MAX_SEQ_NOT_1 2000
#define CONSERVATIVE_GROUP_NUMBER 9
#define SEMI_CONSERVATIVE_GROUP_NUMBER 11
#define SEMI_CONSERVATIVE_GROUP_NUMBER_CHARS 8
#define CONSERVATIVE_GROUP_NUMBER_CHARS 5
#define ROOT 0

struct {
    char seq[MAX_SEQ_1];
    double max_alighment_score;
    int max_offset;
    int hypen_location;
}typedef Sequence;

void printSequence(Sequence seq);
double alignment_score(char *seq1,char *seq2,int offset,int hypen,double w1,double w2,double w3,double w4);
Sequence get_max_sequence_with_omp(char* seq1,char* seq2,double w1,double w2,double w3,double w4);
Sequence* readFromFile(double* w1,double* w2,double* w3,double* w4,int* numberOfSeq,char* seq1);
void printResultsToFile(Sequence* seqs,int numberOfSeq);
int is_consevative(char seq1_char , char seq_other_char);
int is_semi_consevative(char seq1_char , char seq_other_char);
