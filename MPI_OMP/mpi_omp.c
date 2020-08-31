#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include <omp.h>
#include "mpi_omp_header.h"

char conservative_groups[CONSERVATIVE_GROUP_NUMBER][CONSERVATIVE_GROUP_NUMBER_CHARS] = 
{"NDEQ","NEQK","STA","MILV","QHRK","NHQK","FYW","HY","MILF"};
char semi_groups[SEMI_CONSERVATIVE_GROUP_NUMBER][SEMI_CONSERVATIVE_GROUP_NUMBER_CHARS] = 
{"SAG","ATV","CSA","SGND","STPA","STNK","NEQHRK","NDEQHK","SNDEQK","HFY","FVLIM"};

int main(int argc, char *argv[])
{
    // Variables
    int my_rank;       /* rank of process */
    int size;          /* number of processes */
    // SEQS
    int numberOfSeq;
    char seq1[MAX_SEQ_1];
    Sequence* seqsArray;
    // Weights
    double w1 , w2 , w3 , w4;
     /* start up MPI */
    double t1, t2;
    MPI_Init(&argc, &argv);
    t1 = MPI_Wtime();

    /* find out process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //master reads the file
    if (my_rank == ROOT) 
    {
        seqsArray = readFromFile(&w1,&w2,&w3,&w4,&numberOfSeq,seq1);
    }
    // Send to All Proccess the Number of Sequences, Seq1 and Weights 
	MPI_Bcast(&numberOfSeq, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&seq1, MAX_SEQ_1, MPI_CHAR, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&w1, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&w2, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&w3, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&w4, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    // Scattter Sequence to each Proccess 
    Sequence* seq_sended = (Sequence*)malloc(sizeof(Sequence) * (numberOfSeq)/size);
    MPI_Scatter(seqsArray,((numberOfSeq)/size)*sizeof(Sequence),MPI_BYTE,
                seq_sended,((numberOfSeq)/size)*sizeof(Sequence),MPI_BYTE,0,MPI_COMM_WORLD);
    // Each proccess calculate the max Sequence
    int p;
    for(p=0;p<(numberOfSeq)/size;p++)
    {
        seq_sended[p] = get_max_sequence_with_omp(seq1,seq_sended[p].seq,w1,w2,w3,w4);               
    }
    // Gathering Back the results
    MPI_Gather(seq_sended,((numberOfSeq)/size)*sizeof(Sequence),MPI_BYTE,
   		seqsArray,((numberOfSeq)/size)*sizeof(Sequence),MPI_BYTE,0,MPI_COMM_WORLD);
    free(seq_sended);
    if(my_rank == ROOT)
    {
        printResultsToFile(seqsArray,numberOfSeq);
        free(seqsArray);
        t2 = MPI_Wtime();
        printf("MPI_Wtime measured time  to be: %1.2f\n", t2-t1);
        fflush(stdout);
    }
    MPI_Finalize();
}
Sequence get_max_sequence_with_omp(char* seq1,char* seq2,double w1,double w2,double w3,double w4)
{
    // Length of sequences
    int len_seq1;
    len_seq1 = strlen(seq1);
    int len_seq2;
    len_seq2 = strlen(seq2);
    //Max Mutant Variables
    int max_mutant_offset;
    char max_mutant_seq[MAX_SEQ_NOT_1];
    double max_sequence_score = (len_seq2 *-(w4+w3+w2+w1))-1;// Less then any score we can get so we can use max
    double max_mutant_score = (len_seq2 *-(w4+w3+w2+w1))-1;// Less then any score we can get so we can use max
    int max_hypen_location;
    Sequence* max_mutants; // will hold the max Mutants for each thread.
    Sequence max_mutant; // will hold the max Mutant.
    int num_of_threads;
    int x;
    int hypen;
     // Each thread will work on his part and then update max_mutants using Thread num
     // so each thread update max mutant of his work on the right place. 
    #pragma omp parallel shared(max_mutants)
    {
        num_of_threads = omp_get_num_threads();
        max_mutants = (Sequence*)malloc(sizeof(Sequence) *num_of_threads);
        #pragma omp for
        for(hypen=1;hypen<len_seq2+1;hypen++)
        {
            int offset;
            double current_offset_score = (len_seq2 *-(w4+w3+w2+w1))-1;// less then any number we can get
            for(offset=0;offset<len_seq1-(len_seq2);offset++)
            {
               current_offset_score = alignment_score(seq1,seq2, offset,hypen, w1, w2, w3, w4);
               // Checking if current score is greater then the current max.
               // if yes updating the Max Mutants in the currect place of the thread
                if(current_offset_score > max_mutant_score)
                {
                    int thread_num = omp_get_thread_num();
                    max_mutant_score = current_offset_score;
                    max_mutants[thread_num].max_alighment_score = current_offset_score;
                    max_mutants[thread_num].max_offset = offset;
                    max_mutants[thread_num].hypen_location = hypen;
                    strcpy(max_mutants[thread_num].seq,seq2);
                }
            }
        }
    } // End pragma
    // Getting the Max mutant after check the max mutants of each thread
    for(x=0;x<num_of_threads;x++)
    {
        if(max_mutants[x].max_alighment_score > max_sequence_score)
        {
            if(max_mutants[x].max_alighment_score != 0)
            {
                max_sequence_score = max_mutants[x].max_alighment_score;
                max_mutant_offset = max_mutants[x].max_offset;
                max_hypen_location = max_mutants[x].hypen_location;
                strcpy(max_mutant_seq,max_mutants[x].seq);
            }
        }
    }
    free(max_mutants);
    max_mutant.hypen_location = max_hypen_location;
    max_mutant.max_alighment_score = max_sequence_score;
    max_mutant.max_offset = max_mutant_offset;
    strcpy(max_mutant.seq,max_mutant_seq);
    return max_mutant;
}
double alignment_score(char *seq1,char *seq2,int offset,int hypen,double w1,double w2,double w3,double w4)
{
    int NumberOfStars  = 0;
    int NumberOfColons = 0;
    int NumberOfPoints = 0;
    int NumberOfSpaces = 0;
    int x;
    int len_seq2;
    len_seq2 = strlen(seq2);
    double alignment_Score;
    // Calculate Score using reduction so each thread will sum the Numbers
    #pragma omp parallel reduction(+:NumberOfStars,NumberOfColons,NumberOfPoints,NumberOfSpaces)
    {
         NumberOfStars  = 0;
         NumberOfColons = 0;
         NumberOfPoints = 0;
         NumberOfSpaces = 0;
        int y;
        #pragma omp for
        for(x=0;x<len_seq2+1; x++)
        {
            if(x < hypen)
            {
                if(seq1[x+offset] ==  seq2[x])
                {
                    NumberOfStars += 1;
                }
                // Check Consevative
                else if (is_consevative(seq1[x+offset],seq2[x]))
                {
                    NumberOfColons += 1;
                }
                // Check Semi Consevative
                else if (is_semi_consevative(seq1[x+offset],seq2[x]))
                {
                    NumberOfPoints += 1;
                }
                else
                {
                    NumberOfSpaces += 1;
                }
            }
            else if(x > hypen)
            {
                if(seq1[x+offset] ==  seq2[x-1])
                {
                    NumberOfStars += 1;
                }
                // Check Consevative
                else if (is_consevative(seq1[x+offset],seq2[x-1]))
                {
                    NumberOfColons += 1;
                }
                // Check Semi Consevative
                else if (is_semi_consevative(seq1[x+offset],seq2[x-1]))
                {
                    NumberOfPoints += 1;
                }
                else
                {
                    NumberOfSpaces += 1;
                }        
            }
        }
    }
    alignment_Score = w1*NumberOfStars-w2*NumberOfColons-w3*NumberOfPoints-w4*(NumberOfSpaces+1);
    return alignment_Score;
}
Sequence* readFromFile(double* w1,double* w2,double* w3,double* w4,int* numberOfSeq,char seq1[])
{
    FILE *input_file;
    char input[MAX_SEQ_NOT_1];
    int j;

    input_file = fopen(INPUT_FILE_NAME, "r");
    if (input_file == NULL)
    {
        fprintf(stderr, "\nError opening file\n");
        exit(1);
    }
    // Read First Line (Weights)
    fscanf(input_file, "%lf %lf %lf %lf", w1, w2 , w3 , w4);
    // Read Second Line (Seq1)
    fscanf(input_file, "%s", seq1);
    // Read Third Line (Number of Seq)
    fscanf(input_file, "%d", numberOfSeq);
    // Read all Sequences
    Sequence* seqArray = (Sequence*)malloc(sizeof(Sequence) * (*numberOfSeq));
    for(j=0;j<(*numberOfSeq); j++)
    {
        // Read squence
        fscanf(input_file, "%s", seqArray[j].seq);
    }
    fclose(input_file);
    return seqArray;
}
void printResultsToFile(Sequence* seqs,int numberOfSeq)
{
    FILE *output_file;
    output_file = fopen(OUTPUT_FILE_NAME, "w");
    if (output_file == NULL)
    {
        fprintf(stderr, "\nError creating file\n");
        exit(1);
    }
    int i;
    for(i=0;i<numberOfSeq;i++)
    {
        fprintf(output_file,"Seq Number:%d Offset:%d Hypen Location:%d\n",i,seqs[i].max_offset,seqs[i].hypen_location);
    }
   fclose(output_file);

}
void printSequence(Sequence seq)
{
    printf("\nPrint Sequence is: %s\n", seq.seq);
    printf("Score is :%f ", seq.max_alighment_score);
    printf("Hypen is :%d ", seq.hypen_location);
    printf("Offset is :%d \n", seq.max_offset);
}

int is_consevative(char seq1_char , char seq_other_char)
{
    int i;
    for(i=0; i < CONSERVATIVE_GROUP_NUMBER;i++)
    {
       // Check if the Chars are in the group (strchr return null if not find them)
       char* p = strchr(conservative_groups[i],seq1_char);
       char* y = strchr(conservative_groups[i],seq_other_char);
       if (p != NULL && y != NULL)
       {
           return 1;
       }
      p = NULL;
      y = NULL;

    }
    return 0;
}

int is_semi_consevative(char seq1_char , char seq_other_char)
{
    int i;
        for(i=0; i < SEMI_CONSERVATIVE_GROUP_NUMBER;i++)
    {
        // Check if the Chars are in the group (strchr return null if not find them)
       char* p = strchr(semi_groups[i],seq1_char);
       char* y = strchr(semi_groups[i],seq_other_char);
       if (p != NULL && y != NULL)
       {
           return 1;
       }
      p = NULL;
      y = NULL;
    }
    return 0;
}
