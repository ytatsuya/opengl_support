// Easy RS-232C LIBrary "erslib.h"  by I.N.
// OS:Windows 2000/XP/7, Windows Mobile 2003/2005
// Compiler:Visual C++ 2005/2008

// 2002/1/9-1/10 ver.1.0
// 2002/1/21 ver.1.1  extended to 8-Port
// 2002/4/18 ver.1.2  extended to 256-Port
// 2002/7/15 ver.1.2A  debug over COM10
// 2002/7/16 ver.1.3  parity, xonoff, ERS_XoffXon(), dtr,cts,dsr,rts
// 2002/11/27 ver.1.4  int->DWORD
// 2003/10/16 ver.1.5  add ERS_BaudRate(), add ERS_PECHAR
// 2005/01/31          support WindowsCE(eVC++ 4.0), add ERS_Putc(),ERS_Getc()
// 2005/02/02          add ERS_ConfigDialog()
//                     add ERS_Printf(),ERS_Puts(),ERS_Gets(), ERS_WPrintf(),ERS_WPuts(),ERS_WGets()
// 2005/02/03          add ERS_WPutc(), ERS_WGetc()
// 2005/02/08 ver.1.6  updating help file
// 2006/10/20          ers_check() -> int ers_check()
// 2006/12/25 ver.1.7  ERS_Open(),ERS_ConfigDialog() for Visual Studio 2005
// 2013/02/12 ver.1.8  add ERS_Sends(),ERS_WSends()

#pragma once
#include <windows.h>
#include <stdio.h>

//�p���e�B�G���[���̒u����������
#ifndef ERS_PECHAR
#ifdef _WIN32_WCE
#define ERS_PECHAR (L' ')
#else
#define ERS_PECHAR (' ')
#endif
#endif

#ifdef _WIN32_WCE
#define ERSLIBMAXPORT 9
int ers_initdone[ERSLIBMAXPORT] = {};	//0�ŏ�����
#else
#define ERSLIBMAXPORT 256
int ers_initdone[ERSLIBMAXPORT] = {};	//0�ŏ�����
#endif

#define ERS_110			0x00000001
#define ERS_300			0x00000002
#define ERS_600			0x00000003
#define ERS_1200		0x00000004
#define ERS_2400		0x00000005
#define ERS_4800		0x00000006
#define ERS_9600		0x00000007
#define ERS_14400		0x00000008
#define ERS_19200		0x00000009
#define ERS_38400		0x0000000A
#define ERS_56000		0x0000000B
#define ERS_57600		0x0000000C
#define ERS_115200		0x0000000D
#define ERS_128000		0x0000000E
#define ERS_256000		0x0000000F

#define ERS_1			0x00000010
#define ERS_15			0x00000020
#define ERS_2			0x00000030

#define ERS_NO			0x00000100
#define ERS_ODD			0x00000200
#define ERS_EVEN		0x00000300
#define ERS_MARK		0x00000400
#define ERS_SPACE		0x00000500

#define ERS_4			0x00001000
#define ERS_5			0x00002000
#define ERS_6			0x00003000
#define ERS_7			0x00004000
#define ERS_8			0x00005000

#define ERS_DTR_N		0x00010000
#define ERS_DTR_Y		0x00020000
#define ERS_DTR_H		0x00030000

#define ERS_RTS_N		0x00100000
#define ERS_RTS_Y		0x00200000
#define ERS_RTS_H		0x00300000
#define ERS_RTS_T		0x00400000

#define ERS_CTS_Y		0x01000000
#define ERS_CTS_N		0x02000000
#define ERS_DSR_Y		0x04000000
#define ERS_DSR_N		0x08000000

#define ERS_X_Y			0x10000000
#define ERS_X_N			0x20000000

//�n���h���l
HANDLE ers_hcom[ERSLIBMAXPORT];

#define ERSHCOMn (ers_hcom[n-1])

// �p�����[�^ n �`�F�b�N�p�⏕�֐�
// n ���͈͊O�ł�������COMn�����������̏ꍇ�P���߂�D
int ers_check(int n)
{
	if (n<1 || n>ERSLIBMAXPORT) return 1;
	if (!ers_initdone[n - 1]) return 1;
	return 0;
}

//XON/XOFF�������l�̐ݒ�		ver.1.3
// xoff : �o�b�t�@�̎c��T�C�Y�i�o�C�g�P�ʁj
// xon  : �o�b�t�@���̃f�[�^���i�o�C�g�P�ʁj
int ERS_XoffXon(int n, int xoff, int xon)
{
	DCB dcb;

	if (ers_check(n)) return 1;

	GetCommState(ERSHCOMn, &dcb);
	dcb.XoffLim = xoff;
	dcb.XonLim = xon;
	if (!SetCommState(ERSHCOMn, &dcb)) return 2;
	return 0;
}

// �ʐM�p�����[�^�̐ݒ�		ver.1.3
int ERS_Config(int n, unsigned int data)
{
	DCB dcb;
	int d;
	int baud[16] = { 0, CBR_110, CBR_300, CBR_600, CBR_1200, CBR_2400, CBR_4800, CBR_9600, CBR_14400, CBR_19200, CBR_38400, CBR_56000, CBR_57600, CBR_115200, CBR_128000, CBR_256000 };
	int stopbit[4] = { 0, ONESTOPBIT, ONE5STOPBITS, TWOSTOPBITS };
	int parity[6] = { 0, NOPARITY, ODDPARITY, EVENPARITY, MARKPARITY, SPACEPARITY };
	int bytesize[6] = { 0, 4, 5, 6, 7, 8 };
	int dtr[4] = { 0, DTR_CONTROL_DISABLE, DTR_CONTROL_ENABLE, DTR_CONTROL_HANDSHAKE };
	int rts[5] = { 0, RTS_CONTROL_DISABLE, RTS_CONTROL_ENABLE, RTS_CONTROL_HANDSHAKE, RTS_CONTROL_TOGGLE };

	if (ers_check(n)) return 1;

	GetCommState(ERSHCOMn, &dcb);

	// Baud rate
	d = data & 0xF; if (d) dcb.BaudRate = baud[d];

	// Stop bit
	d = (data & 0x30) >> 4;	if (d) dcb.StopBits = stopbit[d];

	// Parity
	d = (data & 0x700) >> 8;
	if (d) {
		dcb.Parity = parity[d];
		if (d>1){
			dcb.fParity = TRUE;
			dcb.fErrorChar = TRUE;
			dcb.ErrorChar = ERS_PECHAR;
		}
		else{
			dcb.fParity = FALSE;
			dcb.fErrorChar = FALSE;
		}
	}

	// Byte size
	d = (data & 0x7000) >> 12; if (d) dcb.ByteSize = bytesize[d];

	// Dtr control
	d = (data & 0x30000) >> 16; if (d) dcb.fDtrControl = dtr[d];

	// Rts control
	d = (data & 0x700000) >> 20; if (d) dcb.fRtsControl = rts[d];

	// CTS control
	if (data &  ERS_CTS_Y) dcb.fOutxCtsFlow = TRUE;
	if (data &  ERS_CTS_N) dcb.fOutxCtsFlow = FALSE;
	// DSR control
	if (data &  ERS_DSR_Y) dcb.fOutxDsrFlow = TRUE;
	if (data &  ERS_DSR_N) dcb.fOutxDsrFlow = FALSE;

	// X control
	d = (data & 0x30000000);
	if (d == ERS_X_Y){
		dcb.fTXContinueOnXoff = FALSE;
		dcb.fOutX = TRUE;
		dcb.fInX = TRUE;
	}
	else if (d == ERS_X_N){
		dcb.fTXContinueOnXoff = FALSE;
		dcb.fOutX = FALSE;
		dcb.fInX = FALSE;
	}

	if (!SetCommState(ERSHCOMn, &dcb)) return 2;
	return 0;
}

// ��M�^�C���A�E�g���Ԃ̐ݒ�(ms�P��)	ver.1.3
int ERS_RecvTimeOut(int n, int rto)
{
	COMMTIMEOUTS ct;

	if (ers_check(n)) return 1;
	GetCommTimeouts(ERSHCOMn, &ct);

	ct.ReadIntervalTimeout = rto;
	ct.ReadTotalTimeoutMultiplier = 0;
	ct.ReadTotalTimeoutConstant = rto;

	if (!SetCommTimeouts(ERSHCOMn, &ct)) return 2;
	return 0;
}

// ���M�^�C���A�E�g���Ԃ̐ݒ�(ms�P��)	ver.1.3
int ERS_SendTimeOut(int n, int sto)
{
	COMMTIMEOUTS ct;

	if (ers_check(n)) return 1;
	GetCommTimeouts(ERSHCOMn, &ct);

	ct.WriteTotalTimeoutMultiplier = 0;
	ct.WriteTotalTimeoutConstant = sto;

	if (!SetCommTimeouts(ERSHCOMn, &ct)) return 2;
	return 0;
}

#ifdef _WIN32_WCE
//==================== �V���A���ʐM�̊J�n ====================
// n=1~9...COM1:~COM9:
int ERS_Open(int n, int recv_size, int send_size)
{
	wchar_t comname[6];

	if (n<1 || n>ERSLIBMAXPORT) return 1;
	if (ers_initdone[n - 1]) return 2;

	wsprintf(comname, L"COM%d:", n);

	ERSHCOMn = CreateFile(comname, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);

	// �I�[�v���Ɏ��s�����Ƃ�
	if (ERSHCOMn == INVALID_HANDLE_VALUE)	return 3;

	//����M�o�b�t�@�̐ݒ�
	SetupComm(ERSHCOMn, recv_size, send_size);

	ers_initdone[n - 1] = 1;

	// �ʐM�̃f�t�H���g�ݒ�
	ERS_Config(n, ERS_9600 | ERS_1 | ERS_NO | ERS_8 | ERS_X_N | ERS_CTS_N | ERS_DSR_N | ERS_DTR_Y | ERS_RTS_Y);
	ERS_RecvTimeOut(n, 1000);
	ERS_SendTimeOut(n, 1000);
	return 0;
}
#else
//==================== �V���A���ʐM�̊J�n ====================
// n=1~256...COM1~COM256         ver.1.7
int ERS_Open(int n, int recv_size, int send_size)
{
	wchar_t comname[11];

	if (n<1 || n>ERSLIBMAXPORT) return 1;
	if (ers_initdone[n - 1]) return 2;

	swprintf_s(comname, 11, L"\\\\.\\COM%d", n);

	ERSHCOMn = CreateFile(comname, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);

	// �I�[�v���Ɏ��s�����Ƃ�
	if (ERSHCOMn == INVALID_HANDLE_VALUE)	return 3;

	//����M�o�b�t�@�̐ݒ�
	SetupComm(ERSHCOMn, recv_size, send_size);

	ers_initdone[n - 1] = 1;

	//�ʐM�̃f�t�H���g�ݒ�
	ERS_Config(n, ERS_9600 | ERS_1 | ERS_NO | ERS_8 | ERS_X_N | ERS_CTS_N | ERS_DSR_N | ERS_DTR_Y | ERS_RTS_Y);
	ERS_RecvTimeOut(n, 1000);
	ERS_SendTimeOut(n, 1000);
	return 0;
}
#endif


// ==================== �V���A���ʐM�̏I�� ====================
int ERS_Close(int n)
{
	if (ers_check(n)) return 1;
	if (!CloseHandle(ERSHCOMn)) return 2;
	ers_initdone[n - 1] = 0;
	return 0;
}

// �V���A���ʐM�̏I���i���ׂĕ���j
void ERS_CloseAll(void)
{
	int n;
	for (n = 1; n <= ERSLIBMAXPORT; n++){
		if (ers_initdone[n - 1]) ERS_Close(n);
	}
}

// ��M�o�b�t�@�̃f�[�^��(�o�C�g)�𒲂ׂ�
int ERS_CheckRecv(int n)
{
	COMSTAT cs;
	DWORD err;
	if (ers_check(n)) return 0;
	ClearCommError(ERSHCOMn, &err, &cs);
	return cs.cbInQue;
}

// ���M�o�b�t�@�̃f�[�^��(�o�C�g)�𒲂ׂ�
int ERS_CheckSend(int n)
{
	COMSTAT cs;
	DWORD err;
	if (ers_check(n)) return 0;
	ClearCommError(ERSHCOMn, &err, &cs);
	return cs.cbOutQue;
}

//�f�[�^��M
int ERS_Recv(int n, void *buf, int size)
{
	DWORD m;
	if (ers_check(n)) return 0;
	ReadFile(ERSHCOMn, buf, size, &m, NULL);
	return m;
}

//�P�o�C�g��M ver.1.6
int ERS_Getc(int n)
{
	DWORD m;
	int c = 0;
	if (ers_check(n)) return 0;
	ReadFile(ERSHCOMn, &c, 1, &m, NULL);
	if (!m)return EOF;
	return c;
}

//�P������M Unicode�� 1.7
wchar_t ERS_WGetc(int n)
{
	DWORD m;
	wchar_t c = 0;
	if (ers_check(n)) return 0;
	ReadFile(ERSHCOMn, &c, sizeof(wchar_t), &m, NULL);
	if (!m)return EOF;
	return c;
}

//�������M ver.1.6
//(�ŏ���\r|\n|\0�܂œǂ�)
int ERS_Gets(int n, char *s, int size)
{
	DWORD m;
	int cnt = 0;
	char c;

	if (ers_check(n)) return 0;
	if (!size) return 0;

	for (;;){
		ReadFile(ERSHCOMn, &c, 1, &m, NULL);
		if (!m || c == '\r' || c == '\n' || c == '\0')break;
		*s++ = c;
		cnt++;
		if (size>0 && cnt >= size - 1)break;
	}
	*s = '\0';
	if (m)cnt++;
	return cnt;
}

//�������M Unicode�� 1.7
//(�ŏ���\r|\n|\0�܂œǂ�)
int ERS_WGets(int n, wchar_t *s, int size)
{
	DWORD m;
	int cnt = 0;
	wchar_t c;

	if (ers_check(n)) return 0;
	if (!size) return 0;

	for (;;){
		ReadFile(ERSHCOMn, &c, sizeof(wchar_t), &m, NULL);
		if (!m || c == L'\r' || c == L'\n' || c == L'\0')break;
		*s++ = c;
		cnt++;
		if (size>0 && cnt >= size - 1)break;
	}
	*s = L'\0';
	if (m)cnt++;
	return cnt;
}

//�f�[�^���M
int ERS_Send(int n, void *buf, int size)
{
	DWORD m;
	if (ers_check(n)) return 0;
	WriteFile(ERSHCOMn, buf, size, &m, NULL);
	return m;
}

//�P�o�C�g���M ver.1.6
int ERS_Putc(int n, int c)
{
	DWORD m;
	if (ers_check(n)) return 0;
	WriteFile(ERSHCOMn, &c, 1, &m, NULL);
	return m;
}

//�P�������M Unicode�� 1.7
int ERS_WPutc(int n, wchar_t c)
{
	DWORD m;
	if (ers_check(n)) return 0;
	WriteFile(ERSHCOMn, &c, sizeof(wchar_t), &m, NULL);
	return m / sizeof(wchar_t);
}

//�����񑗐M ver.1.6
//(�I�[\0��\n�֕ϊ�����)
int ERS_Puts(int n, char *s)
{
	DWORD m;
	char c;
	int cnt = 0;

	if (ers_check(n)) return 0;
	while (c = *s++){
		WriteFile(ERSHCOMn, &c, 1, &m, NULL);
		if (!m)return cnt;
		cnt++;
	}
	c = '\n';
	WriteFile(ERSHCOMn, &c, 1, &m, NULL);
	if (!m)return cnt;
	cnt++;
	return cnt;
}

//�����񑗐M Unicode�� 1.7
//(�I�[\0��\n�֕ϊ�����)
int ERS_WPuts(int n, wchar_t *s)
{
	DWORD m;
	wchar_t c;
	int cnt = 0;

	if (ers_check(n)) return 0;
	while ((c = *s++)){
		WriteFile(ERSHCOMn, &c, sizeof(wchar_t), &m, NULL);
		if (!m)return cnt;
		cnt++;
	}
	c = L'\n';
	WriteFile(ERSHCOMn, &c, sizeof(wchar_t), &m, NULL);
	if (!m)return cnt;
	cnt++;
	return cnt;
}

//COM�|�[�g�ւ�printf() 1.7
int ERS_Printf(int n, char *format, ...)
{
	char buf[256];
	va_list vl;

	va_start(vl, format);
#ifdef _WIN32_WCE
	vsprintf(buf, format, vl);
#else
	vsprintf_s(buf, sizeof(buf), format, vl);
#endif
	return ERS_Puts(n, buf);
}

//COM�|�[�g�ւ�printf() Unicode�� 1.7
int ERS_WPrintf(int n, wchar_t *format, ...)
{
	wchar_t buf[256];
	va_list vl;

	va_start(vl, format);

#ifdef _WIN32_WCE
	vswprintf(buf, format, vl);
#else
	vswprintf_s(buf, 256, format, vl);
#endif

	return ERS_WPuts(n, buf);
}

//��M�o�b�t�@�̃N���A
int ERS_ClearRecv(int n)
{
	if (ers_check(n)) return 1;
	PurgeComm(ERSHCOMn, PURGE_RXCLEAR);
	return 0;
}

//���M�o�b�t�@�̃N���A
int ERS_ClearSend(int n)
{
	if (ers_check(n)) return 1;
	PurgeComm(ERSHCOMn, PURGE_TXCLEAR);
	return 0;
}

//�C�ӂ̃{�[���[�g��ݒ� ver.1.5
int ERS_BaudRate(int n, int baudrate)
{
	DCB dcb;

	if (ers_check(n)) return 1;
	GetCommState(ERSHCOMn, &dcb);
	dcb.BaudRate = baudrate;
	if (!SetCommState(ERSHCOMn, &dcb)) return 2;
	return 0;
}

#ifndef _WIN32_WCE
//�_�C�A���O���g�p���Đݒ肷�� ver.1.7  ��WindowsCE�̂�
//�����FCommConfigDialog()��'\\.\'�������O�ɑΉ����Ȃ��H
int ERS_ConfigDialog(int n)
{
	COMMCONFIG cc;
	DWORD size;
	wchar_t comname[7];

	if (ers_check(n)) return 1;

	swprintf_s(comname, 7, L"COM%d", n);

	GetCommConfig(ERSHCOMn, &cc, &size);
	CommConfigDialog(comname, HWND_DESKTOP, &cc);
	if (!SetCommConfig(ERSHCOMn, &cc, sizeof(cc)))return 2;
	return 0;
}
#endif

//�����񑗐M ver.1.8
// (�I�[\0�̑O�܂ő��M���ďI���)
// �߂�l�F���M����������
int ERS_Sends(int n, char *s)
{
	DWORD m;
	char c;
	int cnt = 0;

	if (ers_check(n)) return 0;
	while (c = *s++){
		WriteFile(ERSHCOMn, &c, 1, &m, NULL);
		if (!m)return cnt;
		cnt++;
	}
	return cnt;
}

//�����񑗐M Unicode�� 1.8
// (�I�[\0�̑O�܂ő��M���ďI���)
// �߂�l�F���M����������
int ERS_WSends(int n, wchar_t *s)
{
	DWORD m;
	wchar_t c;
	int cnt = 0;

	if (ers_check(n)) return 0;
	while ((c = *s++)){
		WriteFile(ERSHCOMn, &c, sizeof(wchar_t), &m, NULL);
		if (!m)return cnt;
		cnt++;
	}
	return cnt;
}
