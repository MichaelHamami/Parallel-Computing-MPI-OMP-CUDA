//#pragma once

// Define VAriable and FUnction so Cuda and Main files will both use them

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
void append(char subject[], const char insert[], int pos);
// float alignment_score(char *seq1,char *seq2,int offset,float w1,float w2,float w3,float w4); // Without Cuda
//void alignment_score(int* score,char *seq1,char *seq2,int offset,float w1,float w2,float w3,float w4);

Sequence* readFromFile(double* w1,double* w2,double* w3,double* w4,int* numberOfSeq,char* seq1);
Sequence* createMutants(Sequence seq);
Sequence get_max_sequence_of_mutants(char* seq1,Sequence* mutants,double w1,double w2,double w3,double w4);
Sequence get_max_sequence_with_omp(char* seq1,char* seq2,double w1,double w2,double w3,double w4);
Sequence get_max_sequence_with_only_cuda(char* seq1,char* seq2,double w1,double w2,double w3,double w4);
void cuda_alignment_score(double* current_offset_score,char *seq1,int len_seq1,char *seq2,int len_seq2,int offset,int hypen,double w1,double w2,double w3,double w4);
void printResultsToFile(Sequence* seqs,int numberOfSeq);
