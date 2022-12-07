///////////////////////////////////////////////////////////////////////////
// Alexandre Graell i Amat                                            
// Simulator Regular SC-LDPC code over the BEC (peeling decoder)
// April 2019
//
// Limited iterations, streaming, doping, square window - Roman Sokolovskii
///////////////////////////////////////////////////////////////////////////

//*************************************************************************
//*****************************    INCLUDE    *****************************
//*************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>


/////////////////DEFINES 
//ConvolutionalCodeMemory
#define Def_dv                     4      // Variable node degree
#define Def_dc                     8     // Check node degree
#define Def_L                      50     // chain length
#define Def_M                      500    // lifting factor

#define Def_CNsPos                 Def_M     // Number of CNs per position
#define Def_VNsPos                 Def_M*2   // Number of VNs per position

#define Def_n                      Def_VNsPos*Def_L                // Code length (number of VNs)

// CIRCULAR BUFFER DOES NOT HAVE TERMINATION
#define CIRCULAR
#undef CIRCULAR

#ifndef CIRCULAR
#define Def_nk                     (Def_L+Def_dv-1)*Def_CNsPos // number of CNs
#else
#define Def_nk                     Def_L * Def_CNsPos   // number of CNs (circular)
#endif
#define Def_MaxNumberBlocksError   1000
#define Def_MaxNumberBlocksSim     1000000



//#define Def_n                      20000 // Code length (number of VNs)
//#define Def_nk                     10400 // number of CNs
//#define Def_VNsPos                 400     // Number of VNs per position
//#define Def_CNsPos                 200     // Number of CNs per position



//SubblockLength
//InformationFrameLength(Need_To_Be_a_Multiple_of_k!)
#define Def_epsIni				   0.475
#define Def_epsDelta			   0.00125

//MaxNumIterTurboDecoding
#define Def_MaxNumIt			   500
//Number Of Simulated Points
#define Def_NUM_POINTS		       18
//MinimumNumberOfFramesInError
#define Def_MinNumberFramesError   1000
#define Def_MinNumberFramesSim     3000
#define Def_MaxNumberFramesSim     1000
//CheckFrames
#define Def_CheckFrames			   100

#define Def_vcunstart                   1


/////////////////DEFINES 
#define PI                        3.14159265358979323846264338327950288419
#define PI2                       6.28318530717958647692528676655900576839
#define CANC          1               // opzione: 1=svuota_i_file 0=appende_in_coda 
#define MAX_FRAMELENGTH				   100000       // maximum frame length


#define LUT_SIZE	64
#define LUT_RANGE	5.5f
#define LUT_FACTOR	(LUT_SIZE/LUT_RANGE)
#define LUT_STEP	(LUT_RANGE/LUT_SIZE)
#define LOG_ZERO	(-32)
#define MAX_LOG		( 32)


int VNdegree[Def_n][Def_dv+1];
int CNdegree[Def_nk][Def_dc+1];
int Lji[Def_n][Def_dv+1];               // Pseudo-posterior probabilities
int Lij[Def_nk][Def_dc+1];
int LLRsChannel[Def_n];
int sizetwoSS[Def_n][2];

char VNerased[Def_n];  // 1 if erased, 0 if not erased after the decision on the VN has been made

// Changes from 0 to 1 once the CN sends a "non-erased" message to a VN for the first time
// If it sends the "resolved" message to several VNs at the same time, then the CN has been
// fully recovered by a "ricochet," so this shouldn't count in the number of deg-1 CNs.
char CNresolved[Def_nk];

int perm_code[Def_CNsPos*Def_dc];

int interleaverCN[Def_L][Def_CNsPos*Def_dc];

int dv;
int dc;
int MaxNumIt;
int InitNumIt;


double ShTh[8]={1.,1.,1.,0.818469,0.772280,0.701780,0.637081,0.581775};

/*************************************************************************/
/*****************    VARIABILI GLOBALI O ESTERNE    *********************/
/*************************************************************************/

//Servers
char nom_fichier[256];
double nb_error[3][1000];

int VectorLength;

int MaxNumIterTurbo;
int MaxNumIterTC;
double rate;

char *filename,*pre;                  /* stringa in cui e' scritto il nome di un file */

int *Z1;							  /* descrizione codificatore */
int nu[10];                               /* numero di memorie */
int nuInner;
int stati;                            /* numero di stati */

double rateINIT;
double Eb_N0dB;

int l_frame;
/*int l_interleaverOutIn;
int l_interleaverSymb;
int l_frameOUCod;
int l_frameOUCodSymb;
int l_frameCod;
int l_frameOutInner;*/

int l_frameINIT;

int* perm;			   /* permutazione */



int PeriodFlipping;
int scelta;                           /* criterio di scelta per stop: 0=FRM_SIMULATI, 1=BIT_ERRATI, 2=FRM_ERRATI */
int numero_frame;                     /* numero di frame da simulare*/
int numero_frame_sim;
int f;                                /* numero frame corrente */
int numero_bit_err;                   /* numero di errori da simulare*/
int numero_frame_err;                 /* numero di frame errati da simulare*/
int users_err;                 /* numero di bit errati */
int count_err;               /* numero di errori per frame */
int frame_err;               /* numero di frame errati */
int frame_errP1;

int block_err;              /* number of blocks in error */
int block_err_exp;          /* number of blocks in error, expurgated */

int users_err_exp;           /* number of bits (users) in error, expurgated */
int frame_err_exp;           /* number of frames in error, expurgated */

int nite;                             /* max numero di iterazioni */

unsigned long int l;                  /* 32 celle LFSR */
unsigned long int l_start;            /* valore iniziale LFSR */

long int vc_un;                       /* variabile intera 31 bit per    */
long int vc_unch;                       /* variabile intera 31 bit per    */
long int vc_unInter;                       /* variabile intera 31 bit per    */
                                      /* la variabile uniforme */
long int vc_un_start;                 /* valore iniziale */

double eb_n0_db;                       /* rapporto segnale/disturbo */
double eb_n0_dbIni;
double eb_n0;
double var,st_dev;                    /* varianza e standard deviation */

double g1,g2;                         /* v.c. gaussiane */

int numero_sim;                       /* numero di simulazioni da eseguire */
double delta_eb_n0_db;                 /* incremento per ogni simulazione */
int sim;                              /* numero simulazione corrente */
int iter;                             /* numero iterazione corrente */

int ncheck;                           /* periodo risultati intermedi */

int framess[200];
int totta;

int testf;
int newt;
int old;
int steps;

double factor;
int bitllr;
int bitext;
int maxtab;
int ic[100000];
int maxq,minq;
int maxqx,minqx;
int RandomInt;
int PerPattLower;
int PerPattUpper;
int PerPattInt;


char PermFileName[400];






void initialize_variables(int *n,int *nk,int L, double *rate,double *ShLm)
{

    int i;
    double sum;
    double dvb,dcb,Lb;
    
l_start = 1;
    
numero_sim = Def_NUM_POINTS;
//printf("Number of simulation points %d \n",numero_sim);

/***************************   NUMERO DI FRAME   *************************/
numero_frame_err = Def_MinNumberFramesError;
numero_frame_sim = Def_MinNumberFramesSim;
    
    MaxNumIt = Def_MaxNumIt;

numero_frame=numero_bit_err=0x7FFFFFFF; /* massimo possibile */
numero_frame = Def_MaxNumberFramesSim;
    
    ncheck = Def_CheckFrames;
    
//Code parameters

*n    = Def_n;
*nk   = Def_nk;
dv = Def_dv;
dc = Def_dc;
  
    dvb = (double) dv;
    dcb = (double) dc;
    Lb  = (double) L;
    sum = 0.;
    
    for(i=1;i<dv;i++)
    {
        sum = sum + 1.-pow((dvb-i)/dvb,dcb);
    }
    
    
    *rate = 1 - (dvb/dcb)*((1 - ((dvb - 1)/Lb)) + 2*(sum/Lb));
    
    
*ShLm = (1.-*rate);
    
    //printf("(%d,%d) ensemble, (L = %d, M = %d) , (n = %d, nk = %d, rate = %f)\n",dv,dc,Def_L,Def_M,*n,*nk,*rate);
    //printf("Shannon limit = %f\n",*ShLm);
    
//printf("load = %f \n",load);

//printf("Minimum number of frame errors to be simulated %d \n",numero_frame_err);
//printf("Maximum number of frames to be simulated %d \n",numero_frame);

/***************************   ITERAZIONI   ******************************/

//printf("Number of iterations %d \n",nite);
    
    

}




double inizio_sim()
{
    int i;
    double eps;
    // inzializzo i contatori degli errori
    
    users_err = 0;
    frame_err = 0;
    frame_errP1 = 0;
    block_err = 0;

    users_err_exp = 0;
    frame_err_exp = 0;
    block_err_exp = 0;

eps  = (double) Def_epsIni - sim*Def_epsDelta;	// Eb/No in dB

l     = l_start;
vc_un = Def_vcunstart;
vc_unch = 37483;
vc_unInter = 1;
    
    for (i=0;i<Def_CNsPos*dc;i++)
    {
        perm_code[i]      = i;
    }

//printf("\n \n \n");
//printf("%s","********************************************************************************");
//printf("Start simulation No. %3d of %3d ( load = %f dB )\n",sim+1,numero_sim,load);

    return(eps);
}

/*************************************************************************/
/******************************   LFSR   *********************************/
/*************************************************************************/

int lfsr()
{
int b,l2;

b = ( (l & (1LL << 31) ) >> 31 ) ;

l2 =   (l & 1) ^ ( (l & (1LL << 1)) >> 1 ) ^ ( (l & (1LL << 21)) >> 21 ) ^ b ;

l = (l << 1) | l2;

return(b);
}




/*************************************************************************/
/*****************************   CANALE    *******************************/
/*************************************************************************/

/*************************************************************************/
/****** unif: generate a uniform distributed random variable *************/
/*************************************************************************/

double unif(void)
{
long s;
long mm=0x7FFFFFFF,a=16807,q=127773,r=2836;

s=vc_un/q;
vc_un=a*(vc_un-q*s)-r*s;
if (vc_un<=0) vc_un+=mm;

return ( (double) vc_un/mm );
}

double unif_ch(void)
{
//    long s;
//    long mm=0x7FFFFFFF,a=16807,q=127773,r=2836;
//
//    s=vc_unch/q;
//    vc_unch=a*(vc_unch-q*s)-r*s;
//    if (vc_unch<=0) vc_unch+=mm;
//
//    return ( (double) vc_unch/mm );
    return (double) random() / RAND_MAX;
}

int unif_int()
{
//long s;
//const long mm=0x7FFFFFFF,a=16807,q=127773,r=2836;
//s=vc_unInter/q;
//vc_unInter=a*(vc_unInter-q*s)-r*s;
//if (vc_unInter<=0) vc_unInter+=mm;
//return (vc_unInter);
return random();
}


/*********************************************************************/
/******* Gaussian: Generate a Gaussian variable with unit variance   */
/*********************************************************************/

double Gaussian()
{
static double f1,f2,f3,a;
static double x2;
static int isready=0;
if(isready)
{
	isready=0;
	return x2;
}
else
{

	f3 = 100.0;
	while (f3 > 1.0)
	{
		f1 = 2.0 * unif() - 1.0 ;
		f2 = 2.0 * unif() - 1.0 ;
		f3 = f1*f1 + f2*f2;
	}
	a = sqrt(-2.0 * log(f3)/f3);  

	x2 = f2 * a;
	isready=1;
	return (f1 * a);
}
}



/*************************************************************************/
/***************************   BER AND FER  ******************************/
/*************************************************************************/

void berfercalculation(int NumErasures)
{
 if (NumErasures>0)
 {
	 frame_err ++;
 }
 //return (count_err);
}









int willIstop()
{
	//if (scelta==2)
	{
		// if (frame_err < numero_frame_err || f < numero_frame_sim) return 0;
        if (frame_err < numero_frame_err) return 0;
	}
	//else if (scelta==0)
	//	if (f < numero_frame) return 0;
	
	return 1;
}


/*************************************************************************/
/***************************   RISULTATI   *******************************/
/*************************************************************************/

void risultati(int sim, int n, int L, int W, double epsilon, double rate, int index)
{
    FILE *fout;
    char stringa[500];
    double pep1;
    double prob;

    /*printf("\nFINAL RESULTS \n");
    printf("Simulation No. %3d of %3d ( Eb/N0 = %f dB )\n",
       sim+1,numero_sim,eb_n0_db);
    printf("Information frame length %d \n",l_frame);
    printf("Last simulated frame No. %d \n",f);
    printf("***************************************************************************\n");
    printf("#iter. Simulated frms   Erred bits        Pb(e)   Erred frms        Pw(e)\n");*/

    pep1 = (double) frame_errP1/f;
    prob = 1.-pow(1.-pep1,Def_L-2);

    // printf("%f %e %e %e %e %e %e\n",epsilon,
    //        (double) users_err/n/f,
    //        (double) frame_err/f,
    //        (double) block_err/L/f,
    //        (double) users_err_exp/n/f,
    //        (double) frame_err_exp/f,
    //        (double) block_err_exp/L/f);
    //printf("%f %e %e %e %e\n",epsilon,(double) users_err/n/f,(double) frame_err/f,(double) frame_errP1/f, prob);

    //sprintf(stringa,"SC_LDPC_(%d,%d)_L%d_M%d_DOP3_BP_SW%d_Random_BLER_%d.dat",Def_dv,Def_dc,Def_L,Def_M,W,index);
    //sprintf(stringa,"SC_LDPC_%d_%d_L%d_M%d_BP_Full_%dit_Random_BLER_%d.dat",Def_dv,Def_dc,Def_L,Def_M,MaxNumIt,index);
    sprintf(stringa,"SC_LDPC_%d_%d_L%d_M%d_BP_SW%d_%dit_%dinit_Random_BLER_%d.dat",Def_dv,Def_dc,Def_L,Def_M,W,MaxNumIt,InitNumIt,index);

    if(sim==0)
    {
        fout = fopen(stringa,"w");
        fprintf(fout,"p BER FER BLER BER_EXP FER_EXP BLER_EXP n L f users_err frame_err block_err users_err_exp frame_err_exp block_err_exp\n");
    }
    else
    {
        fout = fopen(stringa,"a");
    }
    //fprintf(fout,"%f %e %e\n",epsilon,(double) users_err/n/f,(double) frame_err/f);
    fprintf(fout,"%f %e %e %e %e %e %e %d %d %d %d %d %d %d %d %d\n", epsilon,
           (double) users_err/n/f,              // 2
           (double) frame_err/f,                // 3
           (double) block_err/L/f,              // 4
           (double) users_err_exp/n/f,          // 5
           (double) frame_err_exp/f,            // 6
           (double) block_err_exp/L/f,          // 7
           n,               // 8
           L,               // 9
           f,               // 10
           users_err,       // 11
           frame_err,       // 12
           block_err,       // 13
           users_err_exp,   // 14
           frame_err_exp,   // 15
           block_err_exp    // 16
           );
    //fprintf(fout,"%f %e %e %e\n",epsilon,(double) users_err/n/f,(double) frame_err/f,(double) frame_errP1/f);
    //fprintf(fout,"%f %e %e %e %e\n",epsilon,(double) users_err/n/f,(double) frame_err/f,(double) frame_errP1/f, prob);
    fclose(fout);
}


void results_circular(int simulation_point, int n, int L, int W, int num_doped, double epsilon, int index,
    int num_bits_generated, int num_erasures, int num_blocks_generated, int num_blocks_err,
    int num_bits_generated_exp, int num_erasures_exp, int num_blocks_generated_exp, int num_blocks_err_exp)
{
FILE *fout;
char file_name[500];

    printf("%f %e %e %e %e\n", epsilon,
           (double) num_erasures / num_bits_generated,
           (double) num_blocks_err / num_blocks_generated,
           (double) num_erasures_exp / num_bits_generated_exp,
           (double) num_blocks_err_exp / num_blocks_generated_exp);

    sprintf(file_name,"SC_LDPC_%d_%d_L%d_M%d_DOP%d_BP_Stream_SW%d_Random_BLER_%d.dat",Def_dv,Def_dc,Def_L,Def_M,num_doped,W,index);

    if(simulation_point==0)
    {
        fout = fopen(file_name, "w");
        fprintf(fout,"p BER BLER BER_EXP BLER_EXP bit_err bit_gen block_err block_gen bit_err_exp bit_gen_exp block_err_exp block_gen_exp\n");
    }
    else
    {
        fout = fopen(file_name, "a");
    }

    fprintf(fout,"%f %e %e %e %e %d %d %d %d %d %d %d %d\n", epsilon,
            (double) num_erasures / num_bits_generated,              // 2
            (double) num_blocks_err / num_blocks_generated,          // 3
            (double) num_erasures_exp / num_bits_generated_exp,      // 4
            (double) num_blocks_err_exp / num_blocks_generated_exp,  // 5
            num_erasures,              // 6
            num_bits_generated,        // 7
            num_blocks_err,            // 8
            num_blocks_generated,      // 9
            num_erasures_exp,          // 10
            num_bits_generated_exp,    // 11
            num_blocks_err_exp,        // 12
            num_blocks_generated_exp   // 13
    );
    fclose(fout);
}



//////////////////////////////////////////////////////////////////////
//  print_intermediate_results(): Print intermediate results
//////////////////////////////////////////////////////////////////////
void print_intermediate_results(int n, int L, double epsilon)
{
	// printf("\nIntermediate Results (frame no. %d), Sim = %d, frames err = %d\n",(f+1),sim,frame_err);
	// printf("load = %f\n",epsilon);
    // printf("%f %e %e %e %e %e %e\n",epsilon,(double) users_err/n/(f+1),(double) frame_err/(f+1),
    //        (double) block_err / L / (f + 1),
    //        (double) users_err_exp / n / (f + 1),
    //        (double) frame_err_exp / (f + 1),
    //        (double) block_err_exp / L / (f + 1));
}


void read_matrix(int n, int nk)
{
    int i,j;
    int ntemp, nktemp;
    
char stringa[500];
int degree,pos;
FILE *finp;

///Read matrix
sprintf(stringa,"/Users/graell/Box Sync/Alex/Projects/LDPC_Codes/Simulators/LDPC_Simulator_PeelingDecoder_BEC/LDPC_Simulator_PeelingDecoder_BEC/Hmatrix_%d.txt",Def_M);

finp = fopen(stringa, "r");
fscanf(finp,"%d %d\n",&ntemp,&nktemp);
for(i=0;i<ntemp;i++)
    {
        fscanf(finp, "%d ",&degree);
        VNdegree[i][0] = degree;
        for(j=0;j<degree-1;j++)
        {
            fscanf(finp,"%d ",&pos);
            VNdegree[i][1+j] = pos;
        }
        fscanf(finp,"%d\n",&pos);
        VNdegree[i][degree] = pos;
        
    }
    for(i=0;i<nktemp;i++)
    {
        fscanf(finp, "%d ",&degree);
        CNdegree[i][0] = degree;
        for(j=0;j<degree-1;j++)
        {
            fscanf(finp,"%d ",&pos);
            CNdegree[i][1+j] = pos;
        }
        fscanf(finp,"%d\n",&pos);
        CNdegree[i][degree] = pos;
        
    }
    
///Read matrix
    fclose(finp);
}


int decodeBP_SW(int n, int nk, int L, int W, int VNsPos, int CNsPos,
        int *NumErasuresP1, int *num_blocks_err, int *num_erasures_exp, int *num_blocks_err_exp)
{
    // Lji : information that VN j sends to CN i
    // Lij : information that CN i sends to VN j
    
    int i,j,l,m,iter;
    int aux;
    int erasure;
    int NumErasures;
    int NumErasuresPos;
    int NumErasuresTerm;
    int NumErasuresPrecTerm;
    int NumErasuresPrecPos;
    int posW;
    int StartVN,EndVN;
    int StartCN,EndCN;
    int ms = dv-1;

    int NumIt = 0;
    
    // Initialization
    for (j=0; j<n; j++)                        // run over VNs
        for (i=0; i<VNdegree[j][0]; i++)
        {
            Lji[j][i] = LLRsChannel[j];
        }
    for (i=0; i<nk; i++)                        // run over VNs
        for (j=0; j<CNdegree[i][0]; j++)
        {
            Lij[i][j] = 1;
        }
    
    
    
    // iterative processing
    NumErasures = 0;
    *NumErasuresP1 = 0;

    *num_blocks_err     = 0;
    *num_erasures_exp   = 0;
    *num_blocks_err_exp = 0;

    //for(posW=0;posW<L+ms;posW++)
    for(posW=0;posW<L;posW++)
    {
        StartCN = posW*CNsPos;
        EndCN   = StartCN+W*CNsPos;
        if(EndCN>nk) EndCN = nk;
        /* Classical Window */
        // if(posW<=ms)
        // {
        //     StartVN = 0;
        //     EndVN   = StartVN+(W+posW)*VNsPos;
        //     if(EndVN>n) EndVN = n;
        // }
        // else
        // {
        //     StartVN = (posW-ms)*VNsPos;
        //     EndVN   = StartVN+(W+ms)*VNsPos;
        //     if(EndVN>n) EndVN = n;
        // }
        /* Square Window */
        StartVN = posW * VNsPos;
        EndVN = StartVN + W * VNsPos;
        if(EndVN>n) EndVN = n;

        NumErasuresPrecPos = VNsPos;
        iter = 0;                  // Counter of iterations
        NumErasuresPrecTerm = n;

        if (posW == 0)
            NumIt = InitNumIt;
        else
            NumIt = MaxNumIt;

    do
    {
        NumErasuresPos = 0;
        NumErasuresTerm = 0;
        // Check node update
        // Compute outgoing CN messages Lij for each CN (equation (5.20) in Ryan and Lin's book, pp. 222)

        for (i=StartCN; i<EndCN; i++)    // loop over all CNs
            for (j=0; j<CNdegree[i][0]; j++)        // fixing a VN (j) and compute Lij
            {
                erasure = 0;
                for (l=0; l<CNdegree[i][0]; l++)
                {
                    aux = CNdegree[i][1+l];    //Defines the VN participating in the check

                    if (l!=j) // different edge
                    {
                        m = 0;
                        while(VNdegree[aux][1+m]!=i && m<VNdegree[aux][0]) m++;

                        erasure += Lji[aux][m];
                    }
                }

                if(erasure>0) Lij[i][j] = 1;
                else          Lij[i][j] = 0;
            }

        // Variable node update
        // Compute outgoing VN messages Lji for each VN (equation (5.11) in Ryan and Lin's book, pp. 216)

        for (j=StartVN; j<EndVN; j++)
            for (i=0; i<VNdegree[j][0]; i++)
            {
                erasure = 0;

                for (l=0; l<VNdegree[j][0]; l++)
                {
                    aux = VNdegree[j][1+l];

                    if (l!=i)
                    {
                        m = 0;
                        while(CNdegree[aux][1+m]!=j && m<CNdegree[aux][0]) m++;

                        erasure += Lij[aux][m];
                    }
                }

                if (erasure<VNdegree[j][0]-1||LLRsChannel[j]==0) Lji[j][i] = 0;
                else                                             Lji[j][i] = 1;
            }


        // Compute a posteriori LLRs and make decision
        //if(posW>=ms)
        //{
        for (j=StartVN; j<StartVN+VNsPos; j++)
        {
            erasure = 0;
            for (i=0; i<VNdegree[j][0]; i++)
            {
                aux = VNdegree[j][1+i];

                m = 0;
                while(CNdegree[aux][1+m]!=j && m<CNdegree[aux][0]) m++;

                erasure += Lij[aux][m];
            }

            erasure += LLRsChannel[j];

            if (erasure==VNdegree[j][0]+1)
            {
                VNerased[j] = 1;
                NumErasuresPos++;
            }
            else
            {
                VNerased[j] = 0;
            }


        }
        //}

        ///Termination
        for (j=StartVN; j<EndVN; j++)
        {
            erasure = 0;
            for (i=0; i<VNdegree[j][0]; i++)
            {
                aux = VNdegree[j][1+i];

                m = 0;
                while(CNdegree[aux][1+m]!=j && m<CNdegree[aux][0]) m++;

                erasure += Lij[aux][m];
            }

            erasure += LLRsChannel[j];

            if (erasure==VNdegree[j][0]+1) NumErasuresTerm++;


        }

        // printf("pos: %d (VNs %d - %d) iter: %d num_erasures: %d\n", posW, StartVN, EndVN, iter, NumErasuresTerm);

        NumErasuresPrecPos = NumErasuresPos;

        if(NumErasuresTerm==0) break;
        if(NumErasuresTerm == NumErasuresPrecTerm) break;

        NumErasuresPrecTerm = NumErasuresTerm;



        // Stopping criteria
        //stopSPA = check_checks();

        /*if(stopSPA)
         {
         for(i=iter+1;i<MaxNumIterSPA;i++)     bit_error[i][0] += count_err;
         if (count_err)
         for(i=iter+1;i<MaxNumIterSPA;i++) frames_error[i][0]++;
         //printf(" Stopping criterion on (iter=%d)\n",iter);
         break;
         }*/

        // Increment the number of iterations, and check if maximum reached
        iter++;

    //} while (1); // (iter < MaxNumIt);
    //} while (iter < MaxNumIt);
    } while (iter < NumIt);

        NumErasures += NumErasuresPos;

        if (NumErasuresPos > 0)
            *num_blocks_err += 1;

        if(posW>=ms&&posW<=W-2)
            *NumErasuresP1 += NumErasuresPos;
    }

    // Expurgation. Must be done _after_ WD has been completed to filter out just the SS of deg-2.
    // A degree-two SS is a pair of VNs such that:
    //     1. The two VNs are erased
    //     2. They are connected to exactly the same CNs.
    //             => They must be from the same position
    //     3. No other erased VNs are connected to any of the involved CNs.
    // XXX: Extra care should be taken to avoid double-counting.
    for (int vnpos = 0; vnpos < L; vnpos++) {
        int num_erasures_exp_pos = 0;
        for (int a = 0; a < VNsPos; a++) {
            int vn_a = vnpos * VNsPos + a;
            if (VNerased[vn_a]) {
                num_erasures_exp_pos += 1;
                for (int b = a + 1; b < VNsPos; b++) {
                    int vn_b = vnpos * VNsPos + b;
                    if (VNerased[vn_b]) {
                        // => 1. Both VNs are erased
                        int same_connections = 1;
                        int others_recovered = 1;
                        for (int a_conn = 0; a_conn < VNdegree[vn_a][0]; a_conn++)
                        {
                            // index of a CN that vn_a is connected to
                            int aux_a = VNdegree[vn_a][1+a_conn];

                            // for every such CN, vn_b should be connected to it, too
                            int conn_to_vn_b = 0;
                            for (int cn_conn = 0; cn_conn < CNdegree[aux_a][0]; cn_conn++) {
                                int curr_vn = CNdegree[aux_a][1+cn_conn];
                                if (curr_vn == vn_b)
                                    conn_to_vn_b = 1;
                                else if (curr_vn != vn_a && VNerased[curr_vn])
                                    others_recovered = 0;
                            }
                            if (!conn_to_vn_b) {
                                same_connections = 0;
                                break;
                            }
                            if (!others_recovered) {
                                // another erased VN connected to one of the CNs was found.
                                break;
                            }
                        }
                        if (same_connections && others_recovered) {
                            // => 2. vn_a and vn_b are connected to exactly the same CNs
                            // => 3. Other VNs that are connected to these CNs are all recovered
                            // => This is a degree-two stopping set, should be excluded
                            num_erasures_exp_pos -= 2;  // vn_b will be counted as erased later on, too
                            printf("\n\n\nA stopping set of size 2 has been detected! position=%d\n", vnpos);
                        }
                    }
                }
            }
        }
        if (num_erasures_exp_pos > 0) {
            //printf("ERASURE: pos = %d, num = %d\n", vnpos, num_erasures_exp_pos);
            *num_erasures_exp += num_erasures_exp_pos;
            *num_blocks_err_exp += 1;
        }
    }

    return(NumErasures);
    
}


int decodeBP(int n, int nk, int L, int W, int VNsPos, int CNsPos,
                int *num_blocks_err, int *num_erasures_exp, int *num_blocks_err_exp)
{
    // Lji : information that VN j sends to CN i
    // Lij : information that CN i sends to VN j
    
    int i,j,l,m,iter;
    int aux;
    int erasure;
    int NumErasures;
    int NumErasuresPrec = n;
    
    // Initialization
    for (j=0; j<n; j++)                        // run over VNs
        for (i=0; i<VNdegree[j][0]; i++)
        {
            Lji[j][i] = LLRsChannel[j];
        }

    // Truncation: CNs outside of the first L positions are always sending erasure.
    // for (i = Def_L * Def_CNsPos; i < Def_nk; i++)
    //     for (j = 0; j < CNdegree[i][0]; j++)
    //         Lij[i][j] = 1;

    for (int s=0; s < Def_nk; s++)
        CNresolved[s] = 0;

    iter = 0;                  // Counter of iterations

    *num_blocks_err     = 0;
    *num_erasures_exp   = 0;
    *num_blocks_err_exp = 0;
    // iterative processing
    do
    {
        int deg_1_iter = 0;  // the number of degree-one CNs in the current iteration.

        NumErasures = 0;
        // Check node update
        // Compute outgoing CN messages Lij for each CN (equation (5.20) in Ryan and Lin's book, pp. 222)

        // int cn_lim = Def_L * Def_CNsPos;  // truncated -- no CNs after the L-th position
        int cn_lim = nk;
        for (i=0; i<cn_lim; i++)    // loop over all CNs
        {
            int num_out_resolved = 0;
            for (j=0; j<CNdegree[i][0]; j++)        // fixing a VN (j) and compute Lij
            {
                erasure = 0;
                for (l=0; l<CNdegree[i][0]; l++)
                {
                    aux = CNdegree[i][1+l];    //Defines the VN participating in the check
                    
                    if (l!=j) // different edge
                    {
                        m = 0;
                        while(VNdegree[aux][1+m]!=i && m<VNdegree[aux][0]) m++;
                        
                        erasure += Lji[aux][m];
                    }
                }
                
                if(erasure>0) Lij[i][j] = 1;
                else
                {
                    Lij[i][j] = 0;
                    num_out_resolved += 1;
                }
            }
            if (!CNresolved[i])
            {
                if (num_out_resolved > 0)
                {
                    if (num_out_resolved == 1) // The CN was of degree one.
                        deg_1_iter += 1;
                    // num_out_resolved >= 2 means this CN has (was) degree zero
                    CNresolved[i] = 1;
                }
            }
        }
        printf("%d\t%d\t", iter, deg_1_iter);
        
        // Variable node update
        // Compute outgoing VN messages Lji for each VN (equation (5.11) in Ryan and Lin's book, pp. 216)
        
        for (j=0; j<n; j++)
            for (i=0; i<VNdegree[j][0]; i++)
            {
                erasure = 0;
                
                for (l=0; l<VNdegree[j][0]; l++)
                {
                    aux = VNdegree[j][1+l];
                    
                    if (l!=i)
                    {
                        m = 0;
                        while(CNdegree[aux][1+m]!=j && m<CNdegree[aux][0]) m++;
                        
                        erasure += Lij[aux][m];
                    }
                }
                
                if (erasure<VNdegree[j][0]-1||LLRsChannel[j]==0) Lji[j][i] = 0;
                else                                             Lji[j][i] = 1;
            }
        
        
        // Compute a posteriori LLRs and make decision
        for (j=0; j<n; j++)
        {
            erasure = 0;
            for (i=0; i<VNdegree[j][0]; i++)
            {
                aux = VNdegree[j][1+i];
                
                m = 0;
                while(CNdegree[aux][1+m]!=j && m<CNdegree[aux][0]) m++;
                
                erasure += Lij[aux][m];
            }
            
            erasure += LLRsChannel[j];
            
            if (erasure==VNdegree[j][0]+1)
            {
                VNerased[j] = 1;
                NumErasures++;
            } else
            {
                VNerased[j] = 0;
            }

            
        }
        if (deg_1_iter < NumErasuresPrec - NumErasures && iter > 0)
        {
            printf("ARGH! RECOVERED MORE VNs THAN deg-1 CNs! Aborting!\n");
            exit(-1);
        }
        printf("%d\n", NumErasuresPrec - NumErasures);
        
        //berfercalculation(NumErasures);
        
        if(NumErasures==0) break;
        if(NumErasures == NumErasuresPrec) break;
        
        NumErasuresPrec = NumErasures;
        
        // Stopping criteria
        //stopSPA = check_checks();
        
        /*if(stopSPA)
        {
            for(i=iter+1;i<MaxNumIterSPA;i++)     bit_error[i][0] += count_err;
            if (count_err)
                for(i=iter+1;i<MaxNumIterSPA;i++) frames_error[i][0]++;
            //printf(" Stopping criterion on (iter=%d)\n",iter);
            break;
        }*/
        
        // Increment the number of iterations, and check if maximum reached
        iter++;
        
    //} while (1);  // (iter < MaxNumIt);
    } while (iter < MaxNumIt);

    // Expurgation. Must be done _after_ WD has been completed to filter out just the SS of deg-2.
    // A degree-two SS is a pair of VNs such that:
    //     1. The two VNs are erased
    //     2. They are connected to exactly the same CNs.
    //             => They must be from the same position
    //     3. No other erased VNs are connected to any of the involved CNs.
    // XXX: Extra care should be taken to avoid double-counting.
    int is_first_printed = 0;
    for (int vnpos = 0; vnpos < L; vnpos++) {
        int num_erasures_exp_pos = 0;
        int num_erasures_pos = 0;
        for (int a = 0; a < VNsPos; a++) {
            int vn_a = vnpos * VNsPos + a;
            if (VNerased[vn_a]) {
                num_erasures_exp_pos += 1;
                num_erasures_pos += 1;
                for (int b = a + 1; b < VNsPos; b++) {
                    int vn_b = vnpos * VNsPos + b;
                    if (VNerased[vn_b]) {
                        // => 1. Both VNs are erased
                        int same_connections = 1;
                        int others_recovered = 1;
                        for (int a_conn = 0; a_conn < VNdegree[vn_a][0]; a_conn++)
                        {
                            // index of a CN that vn_a is connected to
                            int aux_a = VNdegree[vn_a][1+a_conn];

                            // for every such CN, vn_b should be connected to it, too
                            int conn_to_vn_b = 0;
                            for (int cn_conn = 0; cn_conn < CNdegree[aux_a][0]; cn_conn++) {
                                int curr_vn = CNdegree[aux_a][1+cn_conn];
                                if (curr_vn == vn_b)
                                    conn_to_vn_b = 1;
                                else if (curr_vn != vn_a && VNerased[curr_vn])
                                    others_recovered = 0;
                            }
                            if (!conn_to_vn_b) {
                                same_connections = 0;
                                break;
                            }
                            if (!others_recovered) {
                                // another erased VN connected to one of the CNs was found.
                                break;
                            }
                        }
                        if (same_connections && others_recovered) {
                            // => 2. vn_a and vn_b are connected to exactly the same CNs
                            // => 3. Other VNs that are connected to these CNs are all recovered
                            // => This is a degree-two stopping set, should be excluded
                            num_erasures_exp_pos -= 2;  // vn_b will be counted as erased later on, too
                            printf("\n\n\nA stopping set of size 2 has been detected!\n");
                        }
                    }
                }
            }
        }
        if (num_erasures_pos > 0) {
            *num_blocks_err += 1;
        }
        if (num_erasures_exp_pos > 0 && !is_first_printed) {
            //printf("ERASURE: pos = %d, num = %d\n", vnpos, num_erasures_exp_pos);
            //printf("%d\t%d\n", vnpos, num_erasures_exp_pos);
            is_first_printed = 1;
            *num_erasures_exp += num_erasures_exp_pos;
            *num_blocks_err_exp += 1;
        }
    }
    printf("\n");
    //if (iter == MaxNumIt)
    //    printf("A LIMIT ON THE NUMBER OF ITERATIONS HAS BEEN REACHED\n");

    return(NumErasures);
    
}


/**
 * Initialize the messages to be passed between VNs and CNs
 * based on the values from the channel (for messages from VNs)
 * or on a default value (for messages from CNs).
 * @param pos  absolute position in the chain (not! modulo the size of the circular buffer)
 */
void initialize_messages_circular(int pos, int L, int VNsPos, int CNsPos)
{
    int pos_buffer = pos % L;

    int vn_from = pos_buffer * VNsPos;
    int vn_to = (pos_buffer + 1) * VNsPos;

    int cn_from = pos_buffer * CNsPos;
    int cn_to = (pos_buffer + 1) * CNsPos;

    for (int j = vn_from; j < vn_to; j++)      // run over VNs
        for (int i = 0; i < VNdegree[j][0]; i++)
            Lji[j][i] = LLRsChannel[j];

    for (int i = cn_from; i < cn_to; i++)      // run over CNs
        for (int j = 0; j < CNdegree[i][0]; j++)
            Lij[i][j] = 1;
}


unsigned char calc_sw_range_circular_cn(int pos, int L, int W, int* start_pos_cn, int* end_pos_cn, int* end_pos_cn_wrap)
{
    unsigned char is_wraparound = 0;
    int posW = pos % L;

    *start_pos_cn = posW;
    *end_pos_cn = posW + W;
    *end_pos_cn_wrap = 0;

    if(*end_pos_cn > L)
    {
        is_wraparound = 1;
        *end_pos_cn_wrap = (pos + W) % L;
        *end_pos_cn = L;
    }
    return is_wraparound;
}


unsigned char calc_sw_range_circular_vn(int pos, int L, int W, int ms, int* start_pos_vn, int* end_pos_vn, int* end_pos_vn_wrap)
{
    unsigned char is_wraparound = 0;
    int posW = pos % L;

    *end_pos_vn_wrap = 0;
    if (pos <= ms)
    {
        *start_pos_vn = 0;
        *end_pos_vn = posW + W;
        if (*end_pos_vn > L)
        {
            // EndVN = n;
            printf("ACHTUNG! The window size is greater than the size of the circular buffer (%d vs %d)\nExiting...\n",
                   W, L);
            exit(-1);
        }
    }
    else
    {
        *start_pos_vn = (pos - ms) % L;
        *end_pos_vn = *start_pos_vn + ms + W;
        if (*end_pos_vn > L)
        {
            is_wraparound = 1;
            *end_pos_vn_wrap = (pos + W) % L;
            *end_pos_vn = L;
        }
    }
    return is_wraparound;
}

// Expurgation. Must be done _after_ WD has been completed to filter out just the SS of deg-2.
// A degree-two SS is a pair of VNs such that:
//     1. The two VNs are erased
//     2. They are connected to exactly the same CNs.
//             => They must be from the same position
//     3. No other erased VNs are connected to any of the involved CNs.
// XXX: Extra care should be taken to avoid double-counting.
int get_deg_two_ss(int pos, int VNsPos)
{
    int num_erasures_exp_pos = 0;
    for (int a = 0; a < VNsPos; a++)
    {
        int vn_a = pos * VNsPos + a;
        if (VNerased[vn_a])
        {
            num_erasures_exp_pos += 1;
            for (int b = a + 1; b < VNsPos; b++)
            {
                int vn_b = pos * VNsPos + b;
                if (VNerased[vn_b])
                {
                    // => 1. Both VNs are erased
                    int same_connections = 1;
                    int others_recovered = 1;
                    for (int a_conn = 0; a_conn < VNdegree[vn_a][0]; a_conn++)
                    {
                        // index of a CN that vn_a is connected to
                        int aux_a = VNdegree[vn_a][1 + a_conn];

                        // for every such CN, vn_b should be connected to it, too
                        int conn_to_vn_b = 0;
                        for (int cn_conn = 0; cn_conn < CNdegree[aux_a][0]; cn_conn++)
                        {
                            int curr_vn = CNdegree[aux_a][1 + cn_conn];
                            if (curr_vn == vn_b)
                                conn_to_vn_b = 1;
                            else if (curr_vn != vn_a && VNerased[curr_vn])
                                others_recovered = 0;
                        }
                        if (!conn_to_vn_b)
                        {
                            same_connections = 0;
                            break;
                        }
                        if (!others_recovered)
                        {
                            // another erased VN connected to one of the CNs was found.
                            break;
                        }
                    }
                    if (same_connections && others_recovered)
                    {
                        // => 2. vn_a and vn_b are connected to exactly the same CNs
                        // => 3. Other VNs that are connected to these CNs are all recovered
                        // => This is a degree-two stopping set, should be excluded
                        num_erasures_exp_pos -= 2;  // vn_b will be counted as erased later on, too
                        printf("\n\n\nA stopping set of size 2 has been detected! position=%d\n", pos);
                    }
                }
            }
        }
    }
    return num_erasures_exp_pos;
}

void vn_update(int StartVN, int EndVN)
{
    for (int j = StartVN; j < EndVN; j++)
    {
        for (int i = 0; i < VNdegree[j][0]; i++)
        {
            int erasure = 0;

            for (int k = 0; k < VNdegree[j][0]; k++)
            {
                int aux = VNdegree[j][1 + k];

                if (k != i)
                {
                    int m = 0;
                    while (CNdegree[aux][1 + m] != j && m < CNdegree[aux][0]) m++;

                    erasure += Lij[aux][m];
                }
            }

            if (erasure < VNdegree[j][0] - 1 || LLRsChannel[j] == 0) Lji[j][i] = 0;
            else                                                     Lji[j][i] = 1;
        }
    }
}

void cn_update(int StartCN, int EndCN)
{
    for (int i = StartCN; i < EndCN; i++)    // loop over all CNs
    {
        for (int j = 0; j < CNdegree[i][0]; j++)        // fixing a VN (j) and compute Lij
        {
            int erasure = 0;
            for (int k = 0; k < CNdegree[i][0]; k++)
            {
                int aux = CNdegree[i][1 + k];    //Defines the VN participating in the check

                if (k != j) // different edge
                {
                    int m = 0;
                    while (VNdegree[aux][1 + m] != i && m < VNdegree[aux][0]) m++;

                    erasure += Lji[aux][m];
                }
            }

            if (erasure > 0) Lij[i][j] = 1;
            else             Lij[i][j] = 0;
        }
    }
}


// Compute a posteriori LLRs and make a decision
int calc_erasures_pos(int StartVN, int VNsPos)
{
    int erasures_pos = 0;
    for (int j = StartVN; j < StartVN + VNsPos; j++)
    {
        int erasure = 0;
        for (int i = 0; i < VNdegree[j][0]; i++)
        {
            int aux = VNdegree[j][1 + i];

            int m = 0;
            while (CNdegree[aux][1 + m] != j && m < CNdegree[aux][0]) m++;

            erasure += Lij[aux][m];
        }

        erasure += LLRsChannel[j];

        if (erasure == VNdegree[j][0] + 1)
        {
            VNerased[j] = 1;
            erasures_pos++;
        }
        else
        {
            VNerased[j] = 0;
        }
    }
    return erasures_pos;
}

// Termination. At each iteration, we look over the whole window and decide whether it's time to stop
// iterating and to make a decision on the leftmost position.
int calc_num_erasures_term(int StartVN, int EndVN)
{
    int NumErasuresTerm = 0;
    for (int j = StartVN; j < EndVN; j++)
    {
        int erasure = 0;
        for (int i = 0; i < VNdegree[j][0]; i++)
        {
            int aux = VNdegree[j][1 + i];

            int m = 0;
            while (CNdegree[aux][1 + m] != j && m < CNdegree[aux][0]) m++;

            erasure += Lij[aux][m];
        }

        erasure += LLRsChannel[j];

        if (erasure == VNdegree[j][0] + 1) NumErasuresTerm++;
    }
    return NumErasuresTerm;
}

/**
 * Perform one round of sliding window decoding, slide the window by one position and
 * make the decision on the bits in the leftmost position within the window.
 * NB! The bits we make the decision on are actually at position posW - dv + 1.
 * TODO: NB! Assumes that all required messages (Lij and Lji) are properly initialized via initialize_messages_circular
 * TODO: NB! Assumes that all VNs and CNs are properly connected via generate_code_pos
 */
int decodeBP_SW_circular(int pos, int n, int L, int W, int VNsPos, int CNsPos,
                int *num_blocks_err, int *num_erasures_exp, int *num_blocks_err_exp)
{
    // Lji : information that VN j sends to CN i
    // Lij : information that CN i sends to VN j

    int NumErasuresPos = 0;
    int NumErasuresPrecTerm = 0;
    int ms = dv-1;

    int start_pos_vn, end_pos_vn, end_pos_vn_wrap;
    int start_pos_cn, end_pos_cn, end_pos_cn_wrap;
    calc_sw_range_circular_vn(pos, L, W, ms, &start_pos_vn, &end_pos_vn, &end_pos_vn_wrap);
    calc_sw_range_circular_cn(pos, L, W, &start_pos_cn, &end_pos_cn, &end_pos_cn_wrap);

    int StartVN = start_pos_vn * VNsPos;
    int EndVN = end_pos_vn * VNsPos;
    int EndVN_wrap = end_pos_vn_wrap * VNsPos;  // equals 0 if no wrapping happened

    int StartCN = start_pos_cn * CNsPos;
    int EndCN = end_pos_cn * CNsPos;
    int EndCN_wrap = end_pos_cn_wrap * CNsPos;  // equals 0 if no wrapping happened


    int iteration = 0;                  // Counter of iterations
    NumErasuresPrecTerm = n;

    do
    {
        NumErasuresPos = 0;

        // Check node update
        // Compute outgoing CN messages Lij for each CN (equation (5.20) in Ryan and Lin's book, pp. 222)
        cn_update(StartCN, EndCN);
        cn_update(0, EndCN_wrap);

        // Variable node update
        // Compute outgoing VN messages Lji for each VN (equation (5.11) in Ryan and Lin's book, pp. 216)
        vn_update(StartVN, EndVN);
        vn_update(0, EndVN_wrap);


        if (pos >= ms)
        {
            int erasures_pos = calc_erasures_pos(StartVN, VNsPos);
            NumErasuresPos += erasures_pos;
        }

        int NumErasuresTerm_left = calc_num_erasures_term(StartVN, EndVN);
        int NumErasuresTerm_wrapped = calc_num_erasures_term(0, EndVN_wrap);  // 0 if no wrapping happened
        int NumErasuresTerm = NumErasuresTerm_left + NumErasuresTerm_wrapped;
        // printf("pos: %d (VNs %d - %d) iter: %d num_erasures: %d num_erasures_pos: %d\n", pos, StartVN, EndVN, iteration, NumErasuresTerm, NumErasuresPos);

        if (NumErasuresTerm == 0) break;
        if (NumErasuresTerm == NumErasuresPrecTerm) break;
        // TODO: Can a single VN be left erased with window decoding?
        // {
        //     if (NumErasuresPos == 1)
        //     {
        //         for (int i = 0; i < VNsPos; i++)
        //         {
        //             int VN = (pos % L) * VNsPos + i;
        //             if (VNerased[VN])
        //             {
        //                 printf("ACHTUNG! A SINGLE VN LEFT ERASED!!! %d\n", VN);
        //             }
        //         }
        //     }
        //     break;
        // }

        NumErasuresPrecTerm = NumErasuresTerm;

        iteration++;

    } while (1); // (iteration < MaxNumIt);

    if (NumErasuresPos > 0)
    {
        *num_blocks_err += 1;
    }

    int expurgation_pos = pos - 2 * dv + 1;
    if (expurgation_pos >= 0)
    {
        int expurgation_pos_circular = expurgation_pos % L;

        int num_erasures_exp_pos = get_deg_two_ss(expurgation_pos_circular, VNsPos);
        if (num_erasures_exp_pos > 0)
        {
            printf("ERASURE: pos = %d (%d mod L), num = %d\n", expurgation_pos, expurgation_pos_circular, num_erasures_exp_pos);
            *num_erasures_exp += num_erasures_exp_pos;
            *num_blocks_err_exp += 1;
        }
    }

    return NumErasuresPos;
}


void plr_computation(int NumErasures, int NumErasuresP1,
        int num_blocks_err, int num_erasures_exp, int num_blocks_err_exp)
{
    if(NumErasures > 0)
    {
        users_err += NumErasures;
        frame_err += 1;
        block_err += num_blocks_err;
    }
    if (num_erasures_exp > 0)
    {
        users_err_exp += num_erasures_exp;
        frame_err_exp += 1;
        block_err_exp += num_blocks_err_exp;
    }
    if(NumErasuresP1 > 0)
        frame_errP1 += 1;
}

 //*************************************************************************
//*****************************    MAIN    ********************************
//*************************************************************************
void channel(int n,int nk,double epsilon)
{
    int j;
    double temp;
    
    // BEC
    for (j=0;j<n;j++)
    {
        temp = unif_ch();
        if(temp>=epsilon) //the bit is not erased
        {
            LLRsChannel[j] = 0;
        }
        else
        {
            LLRsChannel[j] = 1;
        }
    }
    
   
}

void channel_doped(int n, double epsilon, int VNsPos, int num_doped, const int doped_positions[])
{
    double temp;

    // BEC
    for (int j = 0; j < n; j++)
    {
        temp = unif_ch();
        if(temp>=epsilon) //the bit is not erased
        {
            LLRsChannel[j] = 0;
        }
        else
        {
            LLRsChannel[j] = 1;
        }
    }

    // DOPING  ->  The bits in doped positions are never erased
    for (int dpos = 0; dpos < num_doped; dpos++)
    {
        int doped_position = doped_positions[dpos];
        for (int j = doped_position * VNsPos; j < (doped_position + 1) * VNsPos; j++)
        {
            LLRsChannel[j] = 0;
        }
    }
}


/**
 * Checks if a given position is doped. The trick is that we assume
 * periodic doping: the number of non-doped positions between doping point is
 * equal to the number of non-doped positions from the beginning of the chain
 * to the first doped position.
 *
 * @param pos  absolute position in the chain (not! modulo the size of the circular buffer)
 * @param num_doped  number of doping points
 * @param doped_positions  the first (!) doping point. We check the location of the leftmost doped position
 * (we assume ascending order of doping positions!) and also the overall "width" of the doping pattern.
 * @return 1 if a position is doped; 0 otherwise.
 */
unsigned char is_position_doped_streaming(int pos, int num_doped, const int doped_positions[])
{
    if (num_doped == 0) return 0;

    int leftmost_doped_position = doped_positions[0];

    int rightmost_doped_position = doped_positions[num_doped - 1];
    int doping_period = rightmost_doped_position + 1;

    int position_mod_doping_period = pos % doping_period;

    if (position_mod_doping_period < leftmost_doped_position)
        return 0;

    for (int id = 0; id < num_doped; id++)
    {
        int dpos = doped_positions[id];
        if (position_mod_doping_period == dpos)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * Roll the channel dice for the circular buffer.
 * Doping is periodic; the number of non-doped positions between doping point is
 * equal to the number of non-doped positions from the beginning of the chain
 * to the first doped position.
 * @param pos  absolute position in the chain (not! modulo the size of the circular buffer)
 */
void generate_channel_doped_circular(int pos, int L, double epsilon, int VNsPos, int num_doped,
                                     const int *doped_positions)
{
    double temp;

    int pos_buffer = pos % L;
    int vn_from =  pos_buffer      * VNsPos;
    int vn_to   = (pos_buffer + 1) * VNsPos;

    int num_erased = 0;

    unsigned char is_doped = is_position_doped_streaming(pos, num_doped, doped_positions);

    if (!is_doped)
    {
        for (int j = vn_from; j < vn_to; j++)
        {
            temp = unif_ch();
            if(temp>=epsilon) // the bit is not erased
                LLRsChannel[j] = 0;
            else
            {
                LLRsChannel[j] = 1;
                num_erased++;
            }
        }
    }
    else
    {
        // bits in doped positions are never erased
        for (int j = vn_from; j < vn_to; j++)
            LLRsChannel[j] = 0;
    }
}

int generate_code(int L, int VNsPos, int CNsPos, int n, int nk)
{
    int i;
    int pos;
    int VN,CN,VNt;
    int ipick,temp;
    int l_interleaver;
    int interleaver[CNsPos*dc];
    int interleaverCN_term[L+dv-1][CNsPos*dc];
    int numSS = 0;

    l_interleaver = CNsPos*dc;
    
    for(i=0;i<nk;i++)
    {
        CNdegree[i][0] = 0;
    }
    
    // for (i=0;i<l_interleaver;i++)
    //     printf("%d ",perm_code[i]);
    // printf("\n");
    
    
    for(pos=0;pos<L+dv-1;pos++)
    {
        //interleave
        for (i=0;i<l_interleaver;i++)
        {
            ipick            = i+unif_int()%(l_interleaver-i);
            temp             = perm_code[i];
            perm_code[i]     = perm_code[ipick];
            perm_code[ipick] = temp;
        }
        for (i=0;i<l_interleaver;i++)
        {
            interleaver[i] = perm_code[i];
            
            CN = pos*CNsPos + (int)((double)interleaver[i]/dc);
            
            interleaverCN_term[pos][i] = CN;
            
            // printf("%d ",interleaver[i]);
        }
        
        // exit(1);
    }
    
    for(pos=0;pos<L;pos++)
    {
        for(VNt=0;VNt<VNsPos;VNt++)
        {
            VN = pos*VNsPos+VNt;
            VNdegree[VN][0] = dv;
            VNerased[VN] = 0; // just initialize with something
            for(i=0;i<dv;i++)
            {
                CN                = interleaverCN_term[pos+i][dv*VNt+i];
                VNdegree[VN][1+i] = CN;
                CNdegree[CN][1+CNdegree[CN][0]] = VN;
                CNdegree[CN][0] = CNdegree[CN][0]+1;
                
                /* if(CNdegree[CN][0]>6)
                 {
                 printf("error CN\n");
                 exit(1);
                 }*/
            }
            
        }
        
    }
    
    //Check size-2 SS
   /* stwoSS = 0;
    for(VN=0;VN<n;VN++)
    {
        fVN = VN+VNsPos*(dv);
        if(fVN>n)fVN = n;
        for(VNb=VN+1;VNb<fVN;VNb++)
        {
            sCNs = 0;
            for(i=0;i<dv;i++)
            {
                for(j=0;j<dv;j++)
                if(VNdegree[VN][1+i]==VNdegree[VNb][1+j])
                {
                    sCNs++;
                }
            }
            if(sCNs==dv)
            {
                stwoSS = 1;
                break;
            }
        }
        if(stwoSS)
        {
            numSS++;
            sizetwoSS[0][0] = numSS;
            sizetwoSS[numSS][0] = VN;
            sizetwoSS[numSS][1] = VNb;
            break;
        }
    }*/
    return(numSS);
}

int fill_interleaver_pos(int pos_buffer, int CNsPos)
{
    int l_interleaver = CNsPos * dc;

    // Create a permutation of CN slots in a single position
    int ipick;
    int temp;
    for (int i = 0; i < l_interleaver; i++)
    {
        ipick            = i + unif_int()%(l_interleaver-i);
        temp             = perm_code[i];
        perm_code[i]     = perm_code[ipick];
        perm_code[ipick] = temp;
    }

    // Fill the interleaver at a position that is dv - 1 steps ahead
    // of the current VN position in the circular buffer.
    for (int i = 0; i < l_interleaver; i++)
    {
        int CN = pos_buffer*CNsPos + (int)((double)perm_code[i]/dc);
        interleaverCN[pos_buffer][i] = CN;
    }

    return 0;
}

int initialize_arrays_circular(int n, int nk, int L, int CNsPos)
{
    for (int i = 0; i < nk; i++)
        CNdegree[i][0] = 0;

    for (int i = 0; i < n; i++)
        for (int j = 0; j <= dv; j++)
            VNdegree[i][j] = 0;

    for (int i = 0; i < n; i++)
        VNerased[i] = 0;

    for (int pos = 0; pos < L; pos++)
    {
        for (int i = 0; i < CNsPos * dc; i++)
        {
            interleaverCN[pos][i] = 0;
        }
    }

    // We need to initialize first dv - 1 positions
    for (int pos = 0; pos < dv - 1; pos++)
    {
        fill_interleaver_pos(pos, CNsPos);
    }
    return 0;
}

/* Generate SC-LDPC code in a streaming fashion using a circular buffer of size L.
 * Effectively, all indexing for positions is done modulo L.
 */
int generate_code_pos(int pos, int L, int VNsPos, int CNsPos)
{
    int i;
    int VN,CN,VNt;

    // VN position that is getting filled
    int pos_buffer = pos % L;
    // CN position that is getting interleaved
    int pos_gen_interleaver = (pos + dv - 1) % L;

    // interleave CNs in the position that is dv - 1 ahead of the current position
    fill_interleaver_pos(pos_gen_interleaver, CNsPos);

    for (int cn = 0; cn < CNsPos; cn++)
    {
        int cn_idx = pos_gen_interleaver * CNsPos + cn;
        // I need to reset it to zero, because we've had some data here before.
        CNdegree[cn_idx][0] = 0;
    }

    // Interconnect the VNs at the specified position 'pos' and the CNs
    // in positions pos to pos + dv - 1.
    for (VNt = 0; VNt < VNsPos; VNt++)
    {
        VN = pos_buffer*VNsPos + VNt;
        VNdegree[VN][0] = dv;
        for(i = 0; i < dv; i++)
        {
            int pos_cn_buffer = (pos_buffer + i) % L;
            CN                = interleaverCN[pos_cn_buffer][dv*VNt+i];
            VNdegree[VN][1+i] = CN;
            CNdegree[CN][1+CNdegree[CN][0]] = VN;
            CNdegree[CN][0] = CNdegree[CN][0]+1;
        }
    }
    return 0;
}

int print_vnmatrix(int L, int VNsPos)
{
    for (int pos = 0; pos < L; pos++)
    {
        printf("pos: %d\n", pos);
        for (int VNt = 0; VNt < VNsPos; VNt++)
        {
            int VN = pos * VNsPos + VNt;
            printf("VN[%d][%d] total: %d (", pos, VNt, VNdegree[VN][0]);
            for (int i = 0; i < dv - 1; i++)
            {
                printf("%d, ", VNdegree[VN][1+i]);
            }
            printf("%d)\n", VNdegree[VN][dv]);
        }
        printf("\n");
    }
    return 0;
}

int test_gen_code_circular(int n, int nk, int L, int VNsPos, int CNsPos)
{
    initialize_arrays_circular(n, nk, L, CNsPos);

    for (int pos = 0; pos < 10; pos++)
    {
        generate_code_pos(pos, L, VNsPos, CNsPos);
        print_vnmatrix(L, VNsPos);
        printf("\n");
    }
    return 0;
}

int test_is_position_doped_streaming()
{
    int num_doped = 3;
    const int doped_positions[] = { 5, 7, 9 };

    for (int pos = 0; pos < 30; pos++)
    {
        unsigned char is_doped = is_position_doped_streaming(pos, num_doped, doped_positions);
        printf("%d\t%d\n", pos, is_doped);
    }

    return 0;
}

int test_circular_buffer_wrapping()
{
    int L = 10;
    int ms = 2;
    int W = 5;

    for (int pos = 0; pos < 40; pos++)
    {
        int start_pos_vn, end_pos_vn, end_pos_vn_wrap;
        int start_pos_cn, end_pos_cn, end_pos_cn_wrap;
        unsigned int is_wrap_vn = calc_sw_range_circular_vn(pos, L, W, ms, &start_pos_vn, &end_pos_vn, &end_pos_vn_wrap);
        unsigned int is_wrap_cn = calc_sw_range_circular_cn(pos, L, W, &start_pos_cn, &end_pos_cn, &end_pos_cn_wrap);

        printf("pos: %d VN:\t%d\t%d\t%d\t%d\t(%d)\n", pos, start_pos_vn, end_pos_vn, 0, end_pos_vn_wrap, is_wrap_vn);
        printf("pos: %d CN:\t%d\t%d\t%d\t%d\t(%d)\n", pos, start_pos_cn, end_pos_cn, 0, end_pos_cn_wrap, is_wrap_cn);
        printf("\n");
    }

    return 0;
}


void generate_stream_pos(int gen_stream_pos, int L, double epsilon, int VNsPos, int CNsPos, int num_doped,
                    const int *doped_positions)
{
    generate_code_pos(gen_stream_pos, L, VNsPos, CNsPos);
    generate_channel_doped_circular(gen_stream_pos, L, epsilon, VNsPos, num_doped, doped_positions);
}

int main_streaming(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Less than 3 arguments! Usage: sw INDEX W NUM_DOPED DOPED_POSITIONS[NUM_DOPED]");
        exit(-1);
    }

    struct timeval te;
    gettimeofday(&te, NULL);
    unsigned long s = te.tv_usec;
    srandom(s);
    int    n;
    int    nk;
    int    L      = Def_L;
    int    CNsPos = Def_CNsPos;
    int    VNsPos = Def_VNsPos;
    double r;
    double ShLm;

    int index = atoi(argv[1]);
    int W     = atoi(argv[2]);
    int num_doped = atoi(argv[3]);

    int doped_positions[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // to be filled later

    int arg_idx = 4;
    for (int i = 0; i < num_doped; i++)
    {
        doped_positions[i] = atoi(argv[arg_idx++]);
    }

    printf("ind=%d W=%d\n", index, W);
    printf("num_doped=%d: {", num_doped);
    for (int i = 0; i < num_doped - 1; i++)
        printf(" %d,", doped_positions[i]);
    if (num_doped == 0)
        printf(" }\n");
    else
        printf(" %d }\n", doped_positions[num_doped - 1]);

    initialize_variables(&n,&nk,L,&r,&ShLm);

    for (sim = 0; sim < numero_sim; sim++)
    {
        int num_erasures = 0;

        int num_blocks_err = 0;
        int num_erasures_exp = 0;
        int num_blocks_err_exp = 0;

        int num_bits_generated = 0;
        int num_blocks_generated = 0;

        int num_bits_generated_exp = 0;
        int num_blocks_generated_exp = 0;

        // 1. Initialize:
        //    - generate sufficient number of VNs and CNs
        //    - generate sufficient number of channel realizations
        // 2. Main loop:
        //    - decode one position
        //    - generate extra position and channel realization in the future
        //    - expurgate one of the previous positions
        //    - update the statistics
        //    - come up with exit criteria (check & exit)

        // the stream is generated stream_lag positions in advance from the current position.
        double epsilon = inizio_sim();
        int stream_lag = L / 2;

        initialize_arrays_circular(n, nk, L, CNsPos);

        int gen_stream_pos = 0;
        for (gen_stream_pos = 0; gen_stream_pos < stream_lag; gen_stream_pos++)
        {
            generate_stream_pos(gen_stream_pos, L, epsilon, VNsPos, CNsPos, num_doped, doped_positions);
            initialize_messages_circular(gen_stream_pos, L, VNsPos, CNsPos);
        }
        // now gen_stream_pos points to the first non-filled position. We'll advance that by one in each position.

        for (int pos = 0 ;; pos++)
        {
            int pos_vn_decision = pos - dv + 1;
            int pos_vn_decision_exp = pos - 2 * dv + 1;
            if (pos_vn_decision >= 0 && !is_position_doped_streaming(pos_vn_decision, num_doped, doped_positions))
            {
                num_bits_generated += VNsPos;
                num_blocks_generated += 1;
            }
            if (pos_vn_decision_exp >= 0 && !is_position_doped_streaming(pos_vn_decision_exp, num_doped, doped_positions))
            {
                num_bits_generated_exp += VNsPos;
                num_blocks_generated_exp += 1;
            }

            num_erasures += decodeBP_SW_circular(pos, n, L, W, VNsPos, CNsPos,
                                                &num_blocks_err, &num_erasures_exp, &num_blocks_err_exp);

            if (num_blocks_err_exp >= Def_MaxNumberBlocksError || num_blocks_generated_exp >= Def_MaxNumberBlocksSim)
                break;

            generate_stream_pos(gen_stream_pos, L, epsilon, VNsPos, CNsPos, num_doped, doped_positions);
            initialize_messages_circular(gen_stream_pos, L, VNsPos, CNsPos);

            if ((pos % 1000) == 0)
            {
                printf("pos=%d (gen_stream_pos=%d) num_erasures=%d num_erasures_exp=%d num_blocks_err=%d num_blocks_err_exp=%d\n",
                       pos, gen_stream_pos, num_erasures, num_erasures_exp, num_blocks_err, num_blocks_err_exp);
            }

            gen_stream_pos++;
        }

        results_circular(sim,n,L,W,num_doped,epsilon,index,
                num_bits_generated,num_erasures,num_blocks_generated,num_blocks_err,
                num_bits_generated_exp,num_erasures_exp,num_blocks_generated_exp,num_blocks_err_exp);
    }

    return 0;
}


int main_terminated(int argc, char *argv[])
{
    struct timeval te;
    gettimeofday(&te, NULL);
unsigned long s = te.tv_usec;
srandom(s);
int    k;
int    n;
int    nk;
int    L      = Def_L;
int    CNsPos = Def_CNsPos;
int    VNsPos = Def_VNsPos;
double rate;
double epsilon;
double ShLm;
int NumErasures;
int NumErasuresPos1 = 0;
int numSS;

int num_blocks_err;
int num_erasures_exp;
int num_blocks_err_exp;

int index = atoi(argv[1]);
int W     = atoi(argv[2]);
int num_doped = atoi(argv[3]);
int max_num_it = atoi(argv[4]);
int init_num_it = atoi(argv[5]);

if (init_num_it == 0)
    init_num_it = max_num_it;

int doped_positions[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // to be filled later

int arg_idx = 4;
for (int i = 0; i < num_doped; i++)
{
    doped_positions[i] = atoi(argv[arg_idx++]);
}

// printf("ind=%d W=%d\n", index, W);
// printf("num_doped=%d: {", num_doped);
// for (int i = 0; i < num_doped - 1; i++)
//     printf(" %d,", doped_positions[i]);
// if (num_doped == 0)
//     printf(" }\n");
// else
//     printf(" %d }\n", doped_positions[num_doped - 1]);

//initialize variables
initialize_variables(&n,&nk,L,&rate,&ShLm);
MaxNumIt = max_num_it;
InitNumIt = init_num_it;
    
k = n-nk;
    
//read_matrix(n,nk);
//compute_distributions(n,nk);

for (sim=0;sim<numero_sim;sim++)
{ // loop on simulation points

    epsilon = inizio_sim();
    //epsilon = 0.6;

		for (f=0;f<numero_frame;f++)
        //for (f=0;f<20000000;f++)
		{ // loop on simulated frames

            //generate code
            numSS = generate_code(L,VNsPos,CNsPos,n,nk);

            if(numSS>0) printf("2-size SS %d\n",f);
            //printf("%d\n",numSS);

	        //channel
          // if(numSS==0)
            {
            // channel(n,nk,epsilon);
            channel_doped(n,epsilon,VNsPos,num_doped,doped_positions);

            //decoding
            //NumErasures = decodeBP(n,nk);
            //NumErasures = decodeBP(n,nk,L,W,VNsPos,CNsPos,
            //        &num_blocks_err, &num_erasures_exp, &num_blocks_err_exp);
            NumErasures = decodeBP_SW(n,nk,L,W,VNsPos,CNsPos,
                    &NumErasuresPos1, &num_blocks_err, &num_erasures_exp, &num_blocks_err_exp);

            plr_computation(NumErasures, NumErasuresPos1, num_blocks_err, num_erasures_exp, num_blocks_err_exp);
            //printf("ONE FRAME SIMULATED\n");

			if (willIstop()) {
				f++; break; }

			if ((f+1)%ncheck==0||NumErasures>0)
            {
                //printf("Num erasures = %d\n",NumErasures);
				print_intermediate_results(n,L,epsilon);
            }
            }

		} // End per tutti i frame

        risultati(sim,n,L,W,epsilon,rate,index);
    
    //exit(1);
} // END loop on simulation points

return 1;
}


int main(int argc, char *argv[])
{
#ifndef CIRCULAR
    main_terminated(argc, argv);
#else
    main_streaming(argc, argv);
#endif
}
